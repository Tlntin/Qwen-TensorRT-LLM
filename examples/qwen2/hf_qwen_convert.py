'''
Convert huggingface QWen-7B-Chat model to numpy file.
Use https://huggingface.co/Qwen/Qwen-7B-Chat as demo.
'''
import argparse
import configparser
import dataclasses
import os
import json
from pathlib import Path

import torch
import torch.multiprocessing as multiprocessing
from smoothquant import capture_activation_range, smooth_gemm, smooth_gemm_mlp
from tqdm import tqdm
from transformers import AutoModelForCausalLM  # transformers-4.10.0-py3
from transformers import AutoTokenizer, GenerationConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
# for debug
from utils.convert import split_and_save_weight
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from default_config import default_config


now_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_ENDPOINT"] = "https://ai.gitee.com/huggingface"
os.environ["HF_HOME"] = "~/.cache/gitee-ai"


@dataclasses.dataclass(frozen=True)
class ProgArgs:
    out_dir: str
    in_file: str
    tensor_parallelism: int = 1
    processes: int = 1
    calibrate_kv_cache: bool = False
    smoothquant: float = None
    model: str = "qwen2"
    storage_type: str = "fp32"
    dataset_cache_dir: str = None

    @staticmethod
    def parse(args=None) -> 'ProgArgs':
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument(
            '--out-dir',
            '-o',
            type=str,
            help='file name of output directory',
            default=default_config.ft_dir_path
            # required=True
        )
        parser.add_argument(
            '--in-file',
            '-i',
            type=str,
            help='file name of input checkpoint file',
            default=default_config.hf_model_dir
            # required=True
        )
        parser.add_argument(
            '--tensor-parallelism',
            '-tp',
            type=int,
            help='Requested tensor parallelism for inference',
            default=1
        )
        parser.add_argument(
            "--processes",
            "-p",
            type=int,
            help=
            "How many processes to spawn for conversion (default: 1). Set it to a lower value to reduce RAM usage.",
            default=1
        )
        parser.add_argument(
            "--calibrate-kv-cache",
            "-kv",
            action="store_true",
            default=None,
            help=
            "Generate scaling factors for KV cache. Used for storing KV cache in int8."
        )
        parser.add_argument(
            "--smoothquant",
            "-sq",
            type=float,
            default=None,
            help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
            " to Smoothquant the model, and output int8 weights."
            " A good first try is 0.5. Must be in [0, 1]")
        parser.add_argument(
            "--model",
            default="qwen2",
            type=str,
            help="Specify GPT variants to convert checkpoints correctly",
            choices=["qwen2", "gpt2", "santacoder", "starcoder"])
        parser.add_argument(
            "--storage-type",
            "-t",
            type=str,
            default="float16",
            choices=["float32", "float16", "bfloat16"]
        )
        parser.add_argument(
            "--dataset-cache-dir",
            type=str,
            default=None,
            help="cache dir to load the hugging face dataset"
        )
        return ProgArgs(**vars(parser.parse_args(args)))


@torch.no_grad()
def smooth_qwen_model(model, scales, alpha, qwen_smoother):
    qwen_qkv_para = {}
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        # qkv/dense in Attention
        if not isinstance(module, Qwen2DecoderLayer):
            continue
        # if isinstance(module, Qwen2Attention):
        layer_name_q = name + ".self_attn.q_proj"
        layer_name_k = name + ".self_attn.k_proj"
        layer_name_v = name + ".self_attn.v_proj"
        layer_name_qkv = name + ".self_attn.qkv"
        qkv_weight = torch.cat(
            [
                module.self_attn.q_proj.weight,
                module.self_attn.k_proj.weight,
                module.self_attn.v_proj.weight
            ],
            dim=0
        )
        smoother = smooth_gemm(
            qkv_weight,
            scales[layer_name_q]["x"],
            module.input_layernorm.weight,
            alpha=alpha
        )
        scales[layer_name_qkv]["x"] = scales[layer_name_q]["x"] / smoother
        scales[layer_name_qkv]["w"] = qkv_weight.abs().max(dim=1)[0]
        scales[layer_name_qkv]["y"] = torch.cat(
            [
                scales[layer_name_q]["y"],
                scales[layer_name_k]["y"],
                scales[layer_name_v]["y"]
            ],
            dim=0
        )
        qwen_qkv_para[layer_name_qkv] = qkv_weight.transpose(0, 1)
        # attention dense
        layer_name = name + ".self_attn.o_proj"
        smoother3 = smooth_gemm(
            module.self_attn.o_proj.weight,
            scales[layer_name]["x"],
            None,
            alpha=alpha,
        )
        qwen_smoother[layer_name] = smoother3.float()

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother3
        scales[layer_name]["w"] = module.self_attn.o_proj.weight.abs().max(dim=1)[0]
        # elif isinstance(module, Qwen2MLP):
        # mlp w1 / w2, because then use some input hidden_states as input, so we need to smooth it with same scale
        mlp_gate_name = name + ".mlp.gate_proj"
        mlp_up_name = name + ".mlp.up_proj"
        smoother2 = smooth_gemm_mlp(
            module.mlp.up_proj.weight,
            module.mlp.gate_proj.weight,
            scales[mlp_up_name]["x"],
            module.post_attention_layernorm.weight,
            alpha=alpha
        )
        scales[mlp_up_name]["x"] = scales[mlp_up_name]["x"] / smoother2
        scales[mlp_gate_name]["x"] = scales[mlp_gate_name]["x"] / smoother2
        scales[mlp_up_name]["w"] = module.mlp.up_proj.weight.abs().max(dim=1)[0]
        scales[mlp_gate_name]["w"] = module.mlp.gate_proj.weight.abs().max(dim=1)[0]

        # mlp down_proj
        layer_name = name + ".mlp.down_proj"
        smoother4 = smooth_gemm(
            module.mlp.down_proj.weight,
            scales[layer_name]["x"],
            None,
            alpha=alpha
        )
        qwen_smoother[layer_name] = smoother4.float()
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother4
        scales[layer_name]["w"] = module.mlp.down_proj.weight.abs().max(dim=1)[0]
    return qwen_qkv_para


# SantaCoder separates Q projection from KV projection
def concat_qkv_weight(q, hf_key, model_params: dict, scales: dict):
    name_q = hf_key
    name_k = hf_key.replace("q_proj", "k_proj")
    name_v = hf_key.replace("q_proj", "v_proj")
    k = model_params[name_k]
    v = model_params[name_v]
    name_q = name_q.replace(".weight", "")
    name_k = name_k.replace(".weight", "")
    name_v = name_v.replace(".weight", "")
    name_qkv = name_q.replace("q_proj", "qkv")
    qkv_weight = torch.cat([q, k, v], dim=0)
    if scales.get(name_q, None) is not None:
        scales[name_qkv]["x"] = scales[name_q]["x"]
        scales[name_qkv]["w"] = qkv_weight.abs().max(dim=1)[0]
        scales[name_qkv]["y"] = torch.cat(
            [
                scales[name_q]["y"],
                scales[name_k]["y"],
                scales[name_v]["y"]
            ],
            dim=0
        )
    return qkv_weight.transpose(0, 1)


def concat_qkv_bias(q, hf_key, model_params: dict):
    k = model_params[hf_key.replace("q_proj", "k_proj")]
    v = model_params[hf_key.replace("q_proj", "v_proj")]
    return torch.cat([q, k, v], dim=0)


# StarCoder uses nn.Linear for these following ops whose weight matrix is transposed compared to transformer.Conv1D
def transpose_weights(hf_name, param):
    weight_to_transpose = [
        "self_attn.qkv", "self_attn.o_proj",
        "mlp.down_proj", "mlp.gate_proj", "mlp.up_proj"
    ]
    if any([k in hf_name for k in weight_to_transpose]):
        if len(param.shape) == 2:
            param = param.transpose(0, 1)
    return param


def qwen_to_ft_name(orig_name):
    global_weights = {
        # 
        "model.embed_tokens.weight": "embed_tokens.weight",
        # "transformer.wpe.weight": "model.wpe",
        # "transformer.ln_f.bias": "model.final_layernorm.bias",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight"
    }

    if orig_name in global_weights:
        return global_weights[orig_name]

    _, _, layer_id, *weight_name = orig_name.split(".")
    layer_id = int(layer_id)
    weight_name = ".".join(weight_name)

    per_layer_weights = {
        "input_layernorm.weight": "input_layernorm.weight",
        "post_attention_layernorm.weight": "post_attention_layernorm.weight",
        "self_attn.q_proj.weight": "self_attn.q_proj.weight",
        "self_attn.q_proj.bias": "self_attn.q_proj.bias",
        "self_attn.k_proj.weight": "self_attn.k_proj.weight",
        "self_attn.k_proj.bias": "self_attn.k_proj.bias",
        "self_attn.v_proj.weight": "self_attn.v_proj.weight",
        "self_attn.v_proj.bias": "self_attn.v_proj.bias",
        "self_attn.o_proj.weight": "self_attn.o_proj.weight",
        "mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "mlp.up_proj.weight": "mlp.up_proj.weight",
        "mlp.down_proj.weight": "mlp.down_proj.weight",
    }
    return f"layers.{layer_id}.{per_layer_weights[weight_name]}"


@torch.no_grad()
def hf_qwen_converter(args: ProgArgs):
    infer_tp = args.tensor_parallelism
    multi_query_mode = True if args.model in ["santacoder", "starcoder"] else False
    saved_dir = Path(args.out_dir) / f"{infer_tp}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    # load position_embedding from rank 0
    model = AutoModelForCausalLM.from_pretrained(
        args.in_file,
        device_map="auto",  # if you gpu memory is not enough, you can set device_map="cpu"
        trust_remote_code=True,
        torch_dtype=str_dtype_to_torch(args.storage_type),
    ).half()  # if you gpu memory is not enough, you can set .half() to .float()
    model.generation_config = GenerationConfig.from_pretrained(
        args.in_file,
        trust_remote_code=True
    )
    act_range = {}
    qwen_smoother = {}
    qwen_qkv_para = {}
    if args.smoothquant is not None or args.calibrate_kv_cache:
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false")
        from datasets import load_dataset
        # copy from summarize.py
        dataset_cnn = load_dataset(
            # "ccdv/cnn_dailymail",
            "cnn_dailymail",
            '3.0.0'
        )
        dataset = dataset_cnn["test"]
        tokenizer = AutoTokenizer.from_pretrained(
            args.in_file,
            legacy=False,
            padding_side='left',
            trust_remote_code=True,
        )
        gen_config_path = os.path.join(args.in_file, 'generation_config.json')
        with open(gen_config_path, 'r') as f:
            gen_config = json.load(f)
        max_input_len = default_config.max_input_len
        # pad_id = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
        # end_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]
        # use this prompt to make chat model do summarize
        system_prompt = "You are a useful assistant, please directly output the corresponding summary according to the article entered by the user."
        act_range = capture_activation_range(
            model,
            tokenizer,
            dataset,
            system_prompt=system_prompt,
            max_input_len=max_input_len,
        )
        if args.smoothquant is not None:
            qwen_qkv_para = smooth_qwen_model(
                model,
                act_range,
                args.smoothquant,
                qwen_smoother
            )

    config = configparser.ConfigParser()
    config["qwen"] = {}
    for key in vars(args):
        config["qwen"][key] = f"{vars(args)[key]}"
    for k, v in vars(model.config).items():
        config["qwen"][k] = f"{v}"
    config["qwen"]["storage_dtype"] = args.storage_type
    config["qwen"]["multi_query_mode"] = str(multi_query_mode)
    with open(saved_dir / "config.ini", 'w') as configfile:
        config.write(configfile)

    storage_type = str_dtype_to_torch(args.storage_type)

    # global_ft_weights = [
    #     "model.wte",
    #     "model.wpe",
    #     "model.final_layernorm.bias",
    #     "model.final_layernorm.weight",
    #     "model.lm_head.weight"
    # ]
    global_ft_weights = [
        "embed_tokens.weight",
        "norm.weight",
        "lm_head.weight",
    ]

    int8_outputs = None
    if args.calibrate_kv_cache:
        int8_outputs = "kv_cache_only"
    if args.smoothquant is not None:
        int8_outputs = "all"

    starmap_args = []
    model_params = dict(model.named_parameters())
    model_params["lm_head.weight"] = model.lm_head.weight
    for name, param in tqdm(
        model_params.items(),
        desc="convert and save",
        total=len(list(model.parameters())),
        ncols=80,
    ):
        if "weight" not in name and "bias" not in name:
            continue
        ft_name = qwen_to_ft_name(name)
        if name.replace(".weight", "") in qwen_smoother.keys():
            smoother = qwen_smoother[name.replace(".weight", "")]
            # smoother = smoother.detach().cpu().numpy()
            starmap_arg = (
                0,
                saved_dir,
                infer_tp,
                f"{ft_name}.smoother".replace(".weight", ""),
                smoother,
                storage_type,
                None,
                {
                     "int8_outputs": int8_outputs,
                     "multi_query_mode": multi_query_mode,
                     "local_dim": None,
                 },
            )
            if args.processes > 1:
                starmap_args.append(starmap_arg)
            else:
                split_and_save_weight(*starmap_arg)

        param = transpose_weights(name, param)
        if ft_name in global_ft_weights:
            torch_to_numpy(param.to(storage_type).cpu()).tofile(
                saved_dir / f"{ft_name}.bin")
        else:
            if 'self_attn.q_proj.weight' in name:
                temp_name = name.replace(".weight", "").replace(
                    ".q_proj",
                    ".qkv"
                )
                if args.smoothquant is None:
                    param = concat_qkv_weight(param, name, model_params, act_range)
                else:
                    param = qwen_qkv_para[temp_name]
                if args.smoothquant or args.calibrate_kv_cache:
                    temp_act_range = act_range[temp_name]
                else:
                    temp_act_range = act_range.get(name.replace(".weight", ""))
                ft_name = ft_name.replace("q_proj", "qkv")
            elif 'self_attn.q_proj.bias' in name:
                param = concat_qkv_bias(param, name, model_params)
                ft_name = ft_name.replace("q_proj", "qkv")
                temp_act_range = act_range.get(name.replace(".weight", ""))
            elif "self_attn.k_proj" in name or "self_attn.v_proj" in name:
                continue
            else:
                temp_act_range = act_range.get(name.replace(".weight", ""))
            # Needed by QKV projection weight split. With multi_query_mode one does not simply take
            # out_dim and divide it by 3 to get local_dim becuase out_dim = local_dim + 2 * head_size
            local_dim = model.model.layers[0].self_attn.q_proj.weight.shape[1] if multi_query_mode else None
            starmap_arg = (
                0,
                saved_dir,
                infer_tp,
                ft_name,
                param.to(storage_type),
                storage_type,
                temp_act_range,
                {
                    "int8_outputs": int8_outputs,
                    "multi_query_mode": multi_query_mode,
                    "local_dim": local_dim
                }
            )
            if args.processes > 1:
                starmap_args.append(starmap_arg)
            else:
                split_and_save_weight(*starmap_arg)

    if args.processes > 1:
        starmap_args = tqdm(starmap_args, desc="saving weights")
        with multiprocessing.Pool(args.processes) as pool:
            pool.starmap(split_and_save_weight, starmap_args)



def run_conversion(args: ProgArgs):
    print("\n=============== Arguments ===============")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("========================================")
    hf_qwen_converter(args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    run_conversion(ProgArgs.parse())
