import time
import configparser
from pathlib import Path
import os
from operator import attrgetter
from safetensors import safe_open
import numpy as np
import torch
from tqdm import tqdm, trange
import tensorrt_llm
from tensorrt_llm._utils import (
    str_dtype_to_torch,
    str_dtype_to_np,
    # pad_vocab_size,
    torch_to_numpy,
)
from tensorrt_llm.quantization import QuantMode
from model import Qwen2ForCausalLM
from tensorrt_llm.mapping import Mapping
from transformers import AutoModelForCausalLM


def gen_suffix(rank, use_smooth_quant, quant_per_channel):
    suffix = f"{rank}.bin"
    if use_smooth_quant:
        sq_prefix = "int8."
        if quant_per_channel:
            sq_prefix += "col."
        suffix = sq_prefix + suffix
    return suffix


def extract_layer_idx(name):
    ss = name.split(".")
    for s in ss:
        if s.isdigit():
            return s
    return None


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    else:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])



def parse_ft_config(ini_file):
    qwen_config = configparser.ConfigParser()
    qwen_config.read(ini_file)

    vocab_size = qwen_config.getint("qwen", "vocab_size")
    hidden_size = qwen_config.getint("qwen", "hidden_size")
    inter_size = qwen_config.getint("qwen", "intermediate_size", fallback=None)
    num_hidden_layers = qwen_config.getint(
        "qwen",
        "num_hidden_layers",
        fallback=32,
    )
    max_position_embeddings = qwen_config.getint(
        "qwen", "max_position_embeddings", fallback=8192
    )
    kv_channels = qwen_config.getint("qwen", "kv_channels", fallback=128)
    rotary_pct = qwen_config.getfloat("qwen", "rotary_pct", fallback=0.0)
    rotary_emb_base = qwen_config.getint("qwen", "rotary_emb_base", fallback=10000)
    multi_query_mode = qwen_config.getboolean(
        "qwen", "multi_query_mode", fallback=False
    )
    return (
        vocab_size,
        hidden_size,
        inter_size,
        num_hidden_layers,
        kv_channels,
        rotary_pct,
        rotary_emb_base,
        multi_query_mode,
        max_position_embeddings,
    )


def load_from_ft(
    tensorrt_llm_qwen: Qwen2ForCausalLM,
    dir_path,
    mapping=Mapping(),
    dtype="float16",
):
    tensorrt_llm.logger.info("Loading weights from FT...")
    tik = time.time()
    quant_mode = getattr(tensorrt_llm_qwen, "quant_mode", QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    (
        vocab_size,
        hidden_size,
        inter_size,
        num_hidden_layers,
        kv_channels,
        rotary_pct,
        rotary_emb_base,
        multi_query_mode,
        max_position_embeddings,
    ) = parse_ft_config(Path(dir_path) / "config.ini")
    np_dtype = str_dtype_to_np(dtype)

    def fromfile(dir_path, name, shape=None, dtype=np.float16):
        dtype = np_dtype if dtype is None else dtype
        p = dir_path + "/" + name
        if Path(p).exists():
            t = np.fromfile(p, dtype=dtype)
            if shape is not None:
                t = t.reshape(shape)
            return t
        else:
            print(f"Warning: {p} not found.")
        return None

    def set_smoothquant_scale_factors(
        module,
        pre_scale_weight,
        dir_path,
        basename,
        shape,
        per_tok_dyn,
        per_channel,
        is_qkv=False,
        rank=None,
    ):
        suffix = "bin"
        if per_channel:
            if rank is not None:
                suffix = f"{rank}." + suffix
            suffix = "col." + suffix

        col_shape = shape if (per_channel or is_qkv) else [1, 1]
        if per_tok_dyn:
            # print(f"{basename}scale_w_quant_orig.{suffix}")
            if pre_scale_weight is not None:
                pre_scale_weight.value = np.array([1.0], dtype=np.float32)
            t = fromfile(
                dir_path,
                f"{basename}scale_w_quant_orig.{suffix}",
                col_shape,
                np.float32,
            )
            module.per_channel_scale.value = t
        else:
            t = fromfile(dir_path, f"{basename}scale_x_orig_quant.bin", [1], np.float32)
            pre_scale_weight.value = t
            t = fromfile(
                dir_path,
                f"{basename}scale_y_accum_quant.{suffix}",
                col_shape,
                np.float32,
            )
            module.per_channel_scale.value = t
            t = fromfile(
                dir_path, f"{basename}scale_y_quant_orig.bin", [1, 1], np.float32
            )
            module.act_scale.value = t

    def set_smoother(module, dir_path, base_name, shape, rank):
        suffix = f"{rank}.bin"
        t = fromfile(dir_path, f"{base_name}.smoother.{suffix}", shape, np.float32)
        module.smoother.value = t

    # Determine the quantization mode.
    quant_mode = getattr(tensorrt_llm_qwen, "quant_mode", QuantMode(0))
    # Do we use SmoothQuant?
    use_smooth_quant = quant_mode.has_act_and_weight_quant()
    # Do we use quantization per token?
    quant_per_token_dyn = quant_mode.has_per_token_dynamic_scaling()
    # Do we use quantization per channel?
    quant_per_channel = quant_mode.has_per_channel_scaling()

    # Do we use INT4/INT8 weight-only?
    use_weight_only = quant_mode.is_weight_only()

    # Int8 KV cache
    use_int8_kv_cache = quant_mode.has_int8_kv_cache()

    # def sq_trick(x):
    #     return x.view(np.float32) if use_smooth_quant else x

    # Debug
    suffix = gen_suffix(mapping.rank, use_smooth_quant, quant_per_channel)
    # The type of weights.
    w_type = np_dtype if not use_smooth_quant else np.int8

    if mapping.is_first_pp_rank():
        tensorrt_llm_qwen.embed_tokens.vocab_embedding.weight.value = fromfile(
            dir_path, "embed_tokens.weight.bin", [vocab_size, hidden_size]
        )

    if mapping.is_last_pp_rank():
        tensorrt_llm_qwen.norm.weight.value = fromfile(
            dir_path, "norm.weight.bin"
        )

    lm_head_weight = fromfile(
        dir_path, "lm_head.weight.bin", [vocab_size, hidden_size]
    )

    if vocab_size % mapping.tp_size != 0:
        # padding
        vocab_size_padded = tensorrt_llm_qwen.lm_head.out_features * mapping.tp_size
        pad_width = vocab_size_padded - vocab_size
        lm_head_weight = np.pad(
            lm_head_weight, ((0, pad_width), (0, 0)), "constant", constant_values=0
        )
    if mapping.is_last_pp_rank():
        tensorrt_llm_qwen.lm_head.weight.value = np.ascontiguousarray(
            split(lm_head_weight, mapping.tp_size, mapping.tp_rank)
        )

    layers_range = trange(
        mapping.pp_rank * tensorrt_llm_qwen.num_layers,
        (mapping.pp_rank + 1) * tensorrt_llm_qwen.num_layers,
        1,
    )

    for i in layers_range:
        c_attn_out_dim = (
            (3 * hidden_size // mapping.tp_size)
            if not multi_query_mode
            else (
                hidden_size // mapping.tp_size + (hidden_size // num_hidden_layers) * 2
            )
        )

        tensorrt_llm_qwen.layers[i].input_layernorm.weight.value = fromfile(
            dir_path, "model.layers." + str(i) + ".input_layernorm.weight.bin"
        )

        dst = tensorrt_llm_qwen.layers[i].post_attention_layernorm.weight
        dst.value = fromfile(
            dir_path,
            "model.layers." + str(i) + ".post_attention_layernorm.weight.bin"
        )
        # self_attn.qkv.weight
        t = fromfile(
            dir_path,
            "model.layers." + str(i) + ".self_attn.qkv.weight." + suffix,
            [hidden_size, c_attn_out_dim],
            w_type,
        )
        if t is not None:
            dst = tensorrt_llm_qwen.layers[i].self_attn.qkv.weight
            if use_smooth_quant:
                dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
                set_smoothquant_scale_factors(
                    tensorrt_llm_qwen.layers[i].self_attn.qkv,
                    tensorrt_llm_qwen.layers[i].input_layernorm.scale_to_int,
                    dir_path,
                    "model.layers." + str(i) + ".self_attn.qkv.",
                    [1, c_attn_out_dim],
                    quant_per_token_dyn,
                    quant_per_channel,
                    rank=mapping.rank,
                    is_qkv=True,
                )
            elif use_weight_only:
                # t = np.ascontiguousarray(np.transpose(t, [1, 0]))
                (
                    processed_torch_weights,
                    torch_weight_scales,
                ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(t), plugin_weight_only_quant_type
                )
                # workaround for trt not supporting int8 inputs in plugins currently
                dst.value = processed_torch_weights.numpy()
                scales = tensorrt_llm_qwen.layers[i].self_attn.qkv.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
        # self_attn.qkv.bias
        dst = tensorrt_llm_qwen.layers[i].self_attn.qkv.bias
        t = fromfile(
            dir_path,
            "model.layers."
            + str(i)
            + ".self_attn.qkv.bias."
            + str(mapping.rank)
            + ".bin",
            [c_attn_out_dim],
        )
        dst.value = np.ascontiguousarray(t)
        # self_attn.o_proj
        dst = tensorrt_llm_qwen.layers[i].self_attn.o_proj.weight
        t = fromfile(
            dir_path,
            "model.layers." + str(i) + ".self_attn.o_proj.weight." + suffix,
            [hidden_size // mapping.tp_size, hidden_size],
            w_type,
        )
        if use_smooth_quant:
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
            dense_scale = getattr(
                tensorrt_llm_qwen.layers[i].self_attn,
                "quantization_scaling_factor",
                None,
            )
            set_smoothquant_scale_factors(
                tensorrt_llm_qwen.layers[i].self_attn.o_proj,
                dense_scale,
                dir_path,
                "model.layers." + str(i) + ".self_attn.o_proj.",
                [1, hidden_size],
                quant_per_token_dyn,
                quant_per_channel,
            )
            set_smoother(
                tensorrt_llm_qwen.layers[i].self_attn.o_proj,
                dir_path,
                "model.layers." + str(i) + ".self_attn.o_proj",
                [1, hidden_size // mapping.tp_size],
                mapping.rank,
            )

        elif use_weight_only:
            # t = np.ascontiguousarray(np.transpose(t, [1, 0]))
            (
                processed_torch_weights,
                torch_weight_scales,
            ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type
            )
            # workaround for trt not supporting int8 inputs in plugins currently
            dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_qwen.layers[i].self_attn.o_proj.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
        # mlp gate_proj
        t = fromfile(
            dir_path,
            "model.layers." + str(i) + ".mlp.gate_proj.weight." + suffix,
            [hidden_size, inter_size // mapping.tp_size],
            w_type,
        )
        if use_smooth_quant:
            tensorrt_llm_qwen.layers[
                i].mlp.gate_proj.weight.value = np.ascontiguousarray(
                np.transpose(t, [1, 0])
            )
            set_smoothquant_scale_factors(
                tensorrt_llm_qwen.layers[i].mlp.gate_proj,
                tensorrt_llm_qwen.layers[i].post_attention_layernorm.scale_to_int,
                dir_path,
                "model.layers." + str(i) + ".mlp.gate_proj.",
                [1, inter_size // mapping.tp_size],
                quant_per_token_dyn,
                quant_per_channel,
                rank=mapping.rank,
            )
        elif use_weight_only:
            dst = tensorrt_llm_qwen.layers[i].mlp.gate_proj.weight
            # t = np.ascontiguousarray(np.transpose(t, [1, 0]))
            (
                processed_torch_weights,
                torch_weight_scales,
            ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type
            )
            # workaround for trt not supporting int8 inputs in plugins currently
            dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_qwen.layers[i].mlp.gate_proj.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_qwen.layers[
                i].mlp.gate_proj.weight.value = np.ascontiguousarray(
                np.transpose(t, [1, 0])
            )
        # mlp up_proj
        t = fromfile(
            dir_path,
            "model.layers." + str(i) + ".mlp.up_proj.weight." + suffix,
            [hidden_size, inter_size // mapping.tp_size],
            w_type,
        )
        if use_smooth_quant:
            tensorrt_llm_qwen.layers[i].mlp.up_proj.weight.value = np.ascontiguousarray(
                np.transpose(t, [1, 0])
            )
            set_smoothquant_scale_factors(
                tensorrt_llm_qwen.layers[i].mlp.up_proj,
                tensorrt_llm_qwen.layers[i].post_attention_layernorm.scale_to_int,
                dir_path,
                "model.layers." + str(i) + ".mlp.up_proj.",
                [1, inter_size // mapping.tp_size],
                quant_per_token_dyn,
                quant_per_channel,
                rank=mapping.rank,
            )
        elif use_weight_only:
            dst = tensorrt_llm_qwen.layers[i].mlp.up_proj.weight
            # t = np.ascontiguousarray(np.transpose(t, [1, 0]))
            (
                processed_torch_weights,
                torch_weight_scales,
            ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type
            )
            # workaround for trt not supporting int8 inputs in plugins currently
            dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_qwen.layers[i].mlp.up_proj.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_qwen.layers[i].mlp.up_proj.weight.value = np.ascontiguousarray(
                np.transpose(t, [1, 0])
            )
        # mlp down_proj
        t = fromfile(
            dir_path,
            "model.layers." + str(i) + ".mlp.down_proj.weight." + suffix,
            [inter_size // mapping.tp_size, hidden_size],
            w_type,
        )
        if use_smooth_quant:
            tensorrt_llm_qwen.layers[i].mlp.down_proj.weight.value = np.ascontiguousarray(
                np.transpose(t, [1, 0])
            )
            proj_scale = getattr(
                tensorrt_llm_qwen.layers[i].mlp, "quantization_scaling_factor", None
            )
            set_smoothquant_scale_factors(
                tensorrt_llm_qwen.layers[i].mlp.down_proj,
                proj_scale,
                dir_path,
                "model.layers." + str(i) + ".mlp.down_proj.",
                [1, hidden_size],
                quant_per_token_dyn,
                quant_per_channel,
            )
            set_smoother(
                tensorrt_llm_qwen.layers[i].mlp.down_proj,
                dir_path,
                "model.layers." + str(i) + ".mlp.down_proj",
                [1, inter_size // mapping.tp_size],
                mapping.rank,
            )
        elif use_weight_only:
            dst = tensorrt_llm_qwen.layers[i].mlp.down_proj.weight
            # t = np.ascontiguousarray(np.transpose(t, [1, 0]))
            (
                processed_torch_weights,
                torch_weight_scales,
            ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type
            )
            # workaround for trt not supporting int8 inputs in plugins currently
            dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_qwen.layers[i].mlp.down_proj.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_qwen.layers[i].mlp.down_proj.weight.value = np.ascontiguousarray(
                np.transpose(t, [1, 0])
            )

        if use_int8_kv_cache:
            t = fromfile(
                dir_path,
                "model.layers." + str(i) + ".self_attn.qkv.scale_y_quant_orig.bin",
                [1],
                np.float32,
            )
            tensorrt_llm_qwen.layers[i].self_attn.kv_orig_quant_scale.value = 1.0 / t
            tensorrt_llm_qwen.layers[i].self_attn.kv_quant_orig_scale.value = t

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f"Weights loaded. Total time: {t}")


def load_from_hf_qwen(
    tensorrt_llm_qwen: Qwen2ForCausalLM,
    hf_qwen,
    mapping=Mapping(),
    # rank=0,
    # tensor_parallel=1,
    max_position_embeddings=8192,
    rotary_base=10000,
    kv_channels=128,
    dtype="float32",
    multi_query_mode=False,
):
    tensorrt_llm.logger.info("Loading weights from HF QWen...")
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_qwen, "quant_mode", QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    model_params = dict(hf_qwen.named_parameters())
    model_params["lm_head.weight"] = hf_qwen.lm_head.weight
    torch_dtype = str_dtype_to_torch(dtype)
    # set for rope embedding
    # inv_freq = 1.0 / (rotary_base ** (
    #     torch.arange(0, kv_channels, 2).float() / kv_channels)
    # )
    # value_table = torch.matmul(
    #     torch.arange(max_position_embeddings).float().reshape(-1, 1),
    #     torch.concat([inv_freq, inv_freq], dim=0).reshape(1, -1)
    # ).reshape(max_position_embeddings, len(inv_freq) * 2)
    # cos_weight = torch.cos(value_table).float()
    # sin_weight = torch.sin(value_table).float()
    # tensorrt_llm_qwen.rope.position_embedding_cos.weight.value = torch_to_numpy(cos_weight)
    # tensorrt_llm_qwen.rope.position_embedding_sin.weight.value = torch_to_numpy(sin_weight)
    for k, v in tqdm(
        model_params.items(), total=len(model_params), ncols=80, desc="Converting..."
    ):
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
        if "model.embed_tokens.weight" in k:
            tensorrt_llm_qwen.embed_tokens.vocab_embedding.weight.value = v
        elif "model.norm.weight" in k:
            tensorrt_llm_qwen.norm.weight.value = v
        elif "lm_head.weight" in k:
            tensorrt_llm_qwen.lm_head.weight.value = np.ascontiguousarray(
                split(v, mapping.tp_size, mapping.rank)
            )
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                print("unknow key: ", k)
                continue
            idx = int(layer_idx)
            if idx >= tensorrt_llm_qwen.num_layers:
                continue
            if "input_layernorm.weight" in k:
                tensorrt_llm_qwen.layers[idx].input_layernorm.weight.value = v
            elif "post_attention_layernorm.weight" in k:
                tensorrt_llm_qwen.layers[idx].post_attention_layernorm.weight.value = v
            elif "self_attn.k_proj.weight" in k or "self_attn.v_proj.weight" in k:
                pass
            elif "self_attn.q_proj.weight" in k:
                dst = tensorrt_llm_qwen.layers[idx].self_attn.qkv.weight
                q_weight = v
                f_str = 'model.layers.{}.self_attn.{}.weight'
                k_weight = model_params[f_str.format(idx, "k_proj")]
                v_weight = model_params[f_str.format(idx, "v_proj")]
                k_weight = torch_to_numpy(k_weight.to(torch_dtype).detach().cpu())
                v_weight = torch_to_numpy(v_weight.to(torch_dtype).detach().cpu())
                if multi_query_mode:
                    wq = split(q_weight, mapping.tp_size, mapping.rank)
                    wk = split(k_weight, mapping.tp_size, mapping.rank)
                    wv = split(v_weight, mapping.tp_size, mapping.rank)
                    split_v = np.concatenate((wq, wk, wv))
                else:
                    q_emb = q_weight.shape[0]
                    model_emb = q_weight.shape[1]
                    qkv_weight = np.concatenate([q_weight, k_weight, v_weight])
                    qkv_weight = qkv_weight.reshape(3, q_emb, model_emb)
                    split_v = split(qkv_weight, mapping.tp_size, mapping.rank, dim=1)
                    split_v = split_v.reshape(3 * (q_emb // mapping.tp_size), model_emb)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    (
                        processed_torch_weights,
                        torch_weight_scales,
                    ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type
                    )
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_qwen.layers[idx].self_attn.qkv.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif "self_attn.k_proj.bias" in k or "self_attn.v_proj.bias" in k:
                pass
            elif "self_attn.q_proj.bias" in k:
                dst = tensorrt_llm_qwen.layers[idx].self_attn.qkv.bias
                q_bias = v
                f_str = 'model.layers.{}.self_attn.{}.bias'
                k_bias = model_params[f_str.format(idx, "k_proj")]
                v_bias = model_params[f_str.format(idx, "v_proj")]
                k_bias = torch_to_numpy(k_bias.to(torch_dtype).detach().cpu())
                v_bias = torch_to_numpy(v_bias.to(torch_dtype).detach().cpu())
                if multi_query_mode:
                    assert isinstance(v, list) and len(v) == 3
                    wq = split(q_bias, mapping.tp_size, mapping.rank)
                    wk = split(k_bias, mapping.tp_size, mapping.rank)
                    wv = split(v_bias, mapping.tp_size, mapping.rank)
                    split_v = np.concatenate((wq, wk, wv))
                else:
                    q_emb = q_bias.shape[0]
                    qkv_bias = np.concatenate([q_bias, k_bias, v_bias])
                    qkv_bias = qkv_bias.reshape(3, q_emb)
                    split_v = split(qkv_bias, mapping.tp_size, mapping.rank, dim=1)
                    split_v = split_v.reshape(3 * (q_emb // mapping.tp_size))
                dst.value = np.ascontiguousarray(split_v)
            elif "self_attn.o_proj.weight" in k:
                dst = tensorrt_llm_qwen.layers[idx].self_attn.o_proj.weight
                split_v = split(v, mapping.tp_size, mapping.rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    (
                        processed_torch_weights,
                        torch_weight_scales,
                    ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type
                    )
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_qwen.layers[
                        idx
                    ].self_attn.o_proj.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif "mlp.gate_proj.weight" in k:
                dst = tensorrt_llm_qwen.layers[idx].mlp.gate_proj.weight
                split_v = split(v, mapping.tp_size, mapping.rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    (
                        processed_torch_weights,
                        torch_weight_scales,
                    ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type
                    )
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_qwen.layers[idx].mlp.gate_proj.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif "mlp.up_proj.weight" in k:
                dst = tensorrt_llm_qwen.layers[idx].mlp.up_proj.weight
                split_v = split(v, mapping.tp_size, mapping.rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    (
                        processed_torch_weights,
                        torch_weight_scales,
                    ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type
                    )
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_qwen.layers[idx].mlp.up_proj.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif "mlp.down_proj.weight" in k:
                dst = tensorrt_llm_qwen.layers[idx].mlp.down_proj.weight
                split_v = split(v, mapping.tp_size, mapping.rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    (
                        processed_torch_weights,
                        torch_weight_scales,
                    ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type
                    )
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_qwen.layers[idx].mlp.down_proj.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            else:
                print("unknow key: ", k)

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f"Weights loaded. Total time: {t}")
    return


def load_from_gptq_qwen(
    tensorrt_llm_qwen: Qwen2ForCausalLM,
    quant_ckpt_path,
    mapping=Mapping(),
    dtype="float16",
):
    tensorrt_llm.logger.info("loading weights from groupwise gptq qwen safetensors...")
    tik = time.time()

    if quant_ckpt_path.endswith(".safetensors"):
        groupwise_qweight_safetensors = safe_open(
            quant_ckpt_path, framework="pt", device="cpu"
        )
        model_params = {
            key: groupwise_qweight_safetensors.get_tensor(key)
            for key in groupwise_qweight_safetensors.keys()
        }
    elif quant_ckpt_path.endswith(".pt"):
        model_params = torch.load(quant_ckpt_path, map_location=torch.device("cpu"))
    else:
        if os.path.isdir(quant_ckpt_path):
            model = AutoModelForCausalLM.from_pretrained(
                quant_ckpt_path,
                device_map="cuda",
                trust_remote_code=True
            ).cpu().eval()
            model_params = {k: v for k, v in model.state_dict().items()}
            torch.cuda.empty_cache()
            del model
        else:
            raise ValueError("quantized checkpoint format not supported!")

    def torch_split(v, dim):
        if v.shape[dim] % mapping.tp_size != 0:
            tensorrt_llm.logger.error(
                "Current weight shape is invalid for mapping.tp_size=" +
                str(mapping.tp_size))
            assert False, "Invalid TP size"
        return v.split(v.shape[dim] // mapping.tp_size,
                       dim=dim)[mapping.tp_rank]

    def unpack_int32_into_int8(w_packed):
        # unpack inputs packed in int32/float32 into uint4 and store them in int8 format
        w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
        w_unpacked = torch.zeros(
            w_packed_int4x2.shape[0], w_packed_int4x2.shape[1] * 2, dtype=torch.int8
        )
        w_unpacked[:, ::2] = w_packed_int4x2 % 16
        w_unpacked[:, 1::2] = w_packed_int4x2 // 16
        return w_unpacked.contiguous()

    def preprocess_groupwise_weight_params(
        weight_name,
        qweight_int32=None,
        qzeros_int32=None,
        scales_fp16=None,
    ):
        if weight_name is not None:
            qweight_int32 = model_params[weight_name].cpu()
            qzeros_int32 = model_params[weight_name[:-7] + "qzeros"].cpu()
            scales_fp16 = model_params[weight_name[:-7] + "scales"].cpu()

        UINT4_TO_INT4_FLAG = 1
        GPTQ_FLAG = 1
        packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
        preprocessor = torch.ops.trtllm.preprocess_weights_for_mixed_gemm

        qweight_unpacked_int8 = (
            unpack_int32_into_int8(qweight_int32.T).T.contiguous() - 8
        ) # qkv weight shape: [4096, 12888], dtype int32 -> uint4x2, save as int8
        qweight_interleaved = preprocessor(
            packer(qweight_unpacked_int8), torch.quint4x2
        ).view(torch.float16) # qkv weight shape: [4096, 4096 * 3]
        # zeros = zeros * scales
        qzeros_unpacked_int32 = unpack_int32_into_int8(qzeros_int32)

        zeros_x_scales_fp16 = (
            -qzeros_unpacked_int32 + 8 * UINT4_TO_INT4_FLAG - GPTQ_FLAG
        ) * scales_fp16
        zeros_x_scales_fp16 = zeros_x_scales_fp16.half()

        # return processed interleaved weight, original scales and zeros * scales
        return (
            qweight_interleaved.contiguous(), # dtype: float32
            zeros_x_scales_fp16.contiguous(), # dtype: float16
            scales_fp16.contiguous(), # dtype: float16
        )

    layer_ids = [extract_layer_idx(key) for key in model_params.keys()]
    layer_ids = [int(layer_idx) for layer_idx in layer_ids if layer_idx is not None]
    num_hidden_layers = max(layer_ids) + 1
    # num_kv_heads = tensorrt_llm_qwen.num_kv_heads
    # mha_mode = num_kv_heads == tensorrt_llm_qwen.num_heads
    suffixs = ["qweight", "qzeros", "scales"]

    layers_per_pipeline_stage = num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(
            mapping.pp_rank * layers_per_pipeline_stage,
            (mapping.pp_rank + 1) * layers_per_pipeline_stage,
            1,
        )
    )
    torch_dtype = str_dtype_to_torch(dtype)
    for layer in tqdm(layers_range, ncols=80, desc="loading attention weight..."):
        idx = layer - mapping.pp_rank * layers_per_pipeline_stage
        # process qkv weight
        prefix = f"model.layers.{layer}.self_attn."
        split_qkv_suf = []
        for suf in suffixs:
            qkv_list = []
            for x in ["q", "k", "v"]:
                x_weight = model_params[prefix + f"{x}_proj." + suf].cpu()
                x_weight = torch_split(x_weight, dim=1)
                qkv_list.append(x_weight)
            split_qkv = torch.cat(qkv_list, dim=1)
            # dype: int32, int32, float16
            split_qkv_suf.append(split_qkv)
        th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
            None,
            split_qkv_suf[0],
            split_qkv_suf[1],
            split_qkv_suf[2],
        )
        tensorrt_llm_qwen.layers[idx].self_attn.qkv.weight.value = th_qweight.numpy()
        tensorrt_llm_qwen.layers[idx].self_attn.qkv.zero.value = th_zero.numpy()
        tensorrt_llm_qwen.layers[idx].self_attn.qkv.weights_scaling_factor.value = th_scale.to(
            torch_dtype).numpy()
        # process qkv bias
        qkv_bias_list = []
        for x in ["q", "k", "v"]:
            x_bias = model_params[prefix + f"{x}_proj.bias"].cpu()
            x_bias = torch_split(x_bias, dim=0)
            qkv_bias_list.append(x_bias)
        qkv_bias = torch.cat(qkv_bias_list, dim=0)

        tensorrt_llm_qwen.layers[idx].self_attn.qkv.bias.value = np.ascontiguousarray(
            qkv_bias.numpy()
        )

    for k, v in tqdm(
            model_params.items(),
            ncols=80,
            desc="loading other weight..."
    ):
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())

        if "model.embed_tokens.weight" in k:
            if mapping.is_first_pp_rank():
                tensorrt_llm.logger.info(f"converting: {k}")
                tensorrt_llm_qwen.embed_tokens.vocab_embedding.weight.value = v
        elif "model.norm.weight" in k:
            if mapping.is_last_pp_rank():
                tensorrt_llm_qwen.norm.weight.value = v
        elif "lm_head.weight" in k:
            tensorrt_llm_qwen.lm_head.weight.value = np.ascontiguousarray(
                split(v, mapping.tp_size, mapping.rank)
            )
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if idx not in layers_range:
                continue
            idx = idx - mapping.pp_rank * layers_per_pipeline_stage

            if "input_layernorm.weight" in k:
                tensorrt_llm_qwen.layers[idx].input_layernorm.weight.value = v
            elif "post_attention_layernorm.weight" in k:
                tensorrt_llm_qwen.layers[idx].post_attention_layernorm.weight.value = v
            elif "self_attn.o_proj.weight" in k:
                split_v_suf = []
                for suf in suffixs:
                    v = model_params[k[:-7] + suf].cpu()
                    split_v = v.split(v.shape[0] // mapping.tp_size, dim=0)[
                        mapping.tp_rank
                    ]
                    split_v_suf.append(split_v)
                th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
                    None, split_v_suf[0], split_v_suf[1], split_v_suf[2]
                )
                tensorrt_llm_qwen.layers[idx].self_attn.o_proj.weight.value = th_qweight.numpy()
                tensorrt_llm_qwen.layers[idx].self_attn.o_proj.zero.value = th_zero.numpy()
                tensorrt_llm_qwen.layers[idx].self_attn.o_proj.weights_scaling_factor.value = th_scale.to(torch_dtype).numpy()
            elif "mlp.gate_proj.weight" in k:
                split_v_suf = []
                for suf in suffixs:
                    v = model_params[k[:-7] + suf].cpu()
                    split_v = v.split(v.shape[1] // mapping.tp_size, dim=1)[
                        mapping.tp_rank
                    ]
                    split_v_suf.append(split_v)
                th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
                    None, split_v_suf[0], split_v_suf[1], split_v_suf[2]
                )
                tensorrt_llm_qwen.layers[idx].mlp.gate_proj.weight.value = th_qweight.numpy()
                tensorrt_llm_qwen.layers[idx].mlp.gate_proj.zero.value = th_zero.numpy()
                tensorrt_llm_qwen.layers[idx].mlp.gate_proj.weights_scaling_factor.value = th_scale.to(torch_dtype).numpy()
            elif "mlp.up_proj.weight" in k:
                split_v_suf = []
                for suf in suffixs:
                    v = model_params[k[:-7] + suf].cpu()
                    split_v = v.split(v.shape[1] // mapping.tp_size, dim=1)[
                        mapping.tp_rank
                    ]
                    split_v_suf.append(split_v)
                th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
                    None, split_v_suf[0], split_v_suf[1], split_v_suf[2]
                )
                tensorrt_llm_qwen.layers[idx].mlp.up_proj.weight.value = th_qweight.numpy()
                tensorrt_llm_qwen.layers[idx].mlp.up_proj.zero.value = th_zero.numpy()
                tensorrt_llm_qwen.layers[idx].mlp.up_proj.weights_scaling_factor.value = th_scale.to(torch_dtype).numpy()
            elif "mlp.down_proj.weight" in k:
                split_v_suf = []
                for suf in suffixs:
                    v = model_params[k[:-7] + suf].cpu()
                    split_v = v.split(v.shape[0] // mapping.tp_size, dim=0)[
                        mapping.tp_rank
                    ]
                    split_v_suf.append(split_v)
                th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
                    None, split_v_suf[0], split_v_suf[1], split_v_suf[2]
                )
                tensorrt_llm_qwen.layers[idx].mlp.down_proj.weight.value = th_qweight.numpy()
                tensorrt_llm_qwen.layers[idx].mlp.down_proj.zero.value = th_zero.numpy()
                tensorrt_llm_qwen.layers[idx].mlp.down_proj.weights_scaling_factor.value = th_scale.to(torch_dtype).numpy()

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f"weights loaded. total time: {t}")


def load_from_awq_qwen(
        tensorrt_llm_qwen: Qwen2ForCausalLM,
        quant_ckpt_path,
        mapping=Mapping(),
        dtype="float16",
        ft_model_dir=None
):
    tensorrt_llm.logger.info(
        'Loading weights from groupwise AWQ Qwen checkpoint...')
    tik = time.time()

    if quant_ckpt_path.endswith(".pt"):
        awq_qwen = torch.load(quant_ckpt_path)
        awq_prefix = "model."
        awq_suffix_list = [
            ".weight",
            ".weight_quantizer._amax",
            ".input_quantizer._pre_quant_scale",
        ]
        awq_key_list = [
            "embed_tokens.weight",  # vocab_embedding
            "lm_head",  # lm_head
            "norm.weight",  # ln_f
            "self_attn.",  # attention.qkv
            "_proj",  # qkv suffix
            "self_attn.o_proj",  # attention.dense
            "mlp.gate_proj",  # mlp.gate_proj
            "mlp.up_proj",  # mlp.up_proj
            "mlp.down_proj",  # mlp.down_proj
            "input_layernorm.weight",  # input_layernorm
            "post_attention_layernorm.weight",  # post_layernorm
        ]
        split_sym = "."

        def load(key):
            if "lm_head" in key:
                v = awq_qwen[key]
            else:
                v = awq_qwen[awq_prefix + key]
            return v

        group_size = load("layers.0.self_attn.o_proj.weight").numel() // load(
            "layers.0.self_attn.o_proj.weight_quantizer._amax").numel()
    else:
        assert False, "Unsupported AWQ quantized checkpoint format"

    quant_mode = getattr(tensorrt_llm_qwen, 'quant_mode', QuantMode(0))
    # Int8 KV cache
    use_int8_kv_cache = quant_mode.has_int8_kv_cache()

    packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.trtllm.preprocess_weights_for_mixed_gemm
    torch_dtype = str_dtype_to_torch(dtype)

    def fromfile(dir_path, name, shape=None, dtype=None):
        p = dir_path + '/' + name
        if Path(p).exists():
            t = np.fromfile(p, dtype=dtype)
            if shape is not None:
                t = t.reshape(shape)
            return t
        return None

    def torch_split(v, dim):
        if v.shape[dim] % mapping.tp_size != 0:
            tensorrt_llm.logger.error(
                "Current weight shape is invalid for mapping.tp_size=" +
                str(mapping.tp_size))
            assert False, "Invalid TP size"
        return v.split(v.shape[dim] // mapping.tp_size,
                       dim=dim)[mapping.tp_rank]

    def AWQ_quantize_pack_preprocess(weight, scale):
        weight /= scale.repeat_interleave(group_size, dim=0)
        qweight_int8 = torch.clamp(torch.round(weight.cuda()).char(), -8, 7)
        int4_weight = preprocessor(packer(qweight_int8.cpu()), torch.quint4x2)
        return int4_weight.view(torch.int8).cpu().numpy()

    def process_and_assign_weight(mOp, v, tp_dim=0):
        weight = v[0].T.contiguous()
        [k, n] = weight.shape
        weight = torch_split(weight, tp_dim)
        amax = v[1].reshape((n, k // group_size)).T.contiguous()
        amax = torch_split(amax, tp_dim)
        pre_quant_scale = v[2].reshape((1, k))
        if tp_dim == 0:
            pre_quant_scale = torch_split(pre_quant_scale, 1)
        scale = amax / 8.0
        mOp.qweight.value = AWQ_quantize_pack_preprocess(weight, scale)
        mOp.scale.value = scale.to(torch_dtype).cpu().numpy()
        mOp.pre_quant_scale.value = pre_quant_scale.to(
            torch_dtype).cpu().numpy()

    def reSmooth_and_get_scale(weight, pre_quant_scale, avg_pre_quant_scale):
        # deSmooth and reSmooth
        [k, n] = weight.shape
        if quant_ckpt_path.endswith("pt"):
            # NPZ files are already re-smoothed
            weight *= pre_quant_scale.repeat((n, 1)).transpose(1,
                                                               0).contiguous()
            weight /= avg_pre_quant_scale.repeat(
                (n, 1)).transpose(1, 0).contiguous()

        # Get scale
        weight_t = weight.T.contiguous()
        weight_t = weight_t.reshape(n, k // group_size, group_size)
        weight_t = torch.abs(weight_t.reshape(-1, group_size))
        amax, idx = weight_t.max(1)
        amax = amax.reshape(n, k // group_size).T.contiguous()
        scale = amax / 8
        return weight, scale

    def process_and_assign_qkv_weight(prefix, mOp):
        q_weight = load(prefix + "q" + awq_key_list[4] +
                        awq_suffix_list[0]).T.contiguous()
        k_weight = load(prefix + "k" + awq_key_list[4] +
                        awq_suffix_list[0]).T.contiguous()
        v_weight = load(prefix + "v" + awq_key_list[4] +
                        awq_suffix_list[0]).T.contiguous()
        dim_k = q_weight.shape[0]
        q_weight = torch_split(q_weight, 1)
        k_weight = torch_split(k_weight, 1)
        v_weight = torch_split(v_weight, 1)
        q_pre_quant_scale = load(prefix + "q" + awq_key_list[4] +
                                 awq_suffix_list[2]).reshape((1, dim_k))
        k_pre_quant_scale = load(prefix + "k" + awq_key_list[4] +
                                 awq_suffix_list[2]).reshape((1, dim_k))
        v_pre_quant_scale = load(prefix + "v" + awq_key_list[4] +
                                 awq_suffix_list[2]).reshape((1, dim_k))
        qkv_pre_quant_scale = (q_pre_quant_scale + k_pre_quant_scale +
                               v_pre_quant_scale) / 3.0
        q_weight, q_scale = reSmooth_and_get_scale(q_weight, q_pre_quant_scale,
                                                   qkv_pre_quant_scale)
        k_weight, k_scale = reSmooth_and_get_scale(k_weight, k_pre_quant_scale,
                                                   qkv_pre_quant_scale)
        v_weight, v_scale = reSmooth_and_get_scale(v_weight, v_pre_quant_scale,
                                                   qkv_pre_quant_scale)
        qkv_weights = torch.cat((q_weight, k_weight, v_weight), dim=1)
        qkv_scale = torch.cat((q_scale, k_scale, v_scale), dim=1)

        mOp.pre_quant_scale.value = qkv_pre_quant_scale.to(
            torch_dtype).cpu().numpy()
        mOp.qweight.value = AWQ_quantize_pack_preprocess(qkv_weights, qkv_scale)
        mOp.scale.value = qkv_scale.to(torch_dtype).cpu().numpy()

    # Load weights from AWQ checkpoint into TRT-LLM module
    # 1. vocab_embedding
    v = load(awq_key_list[0])
    # TRT-LLM requires vocab_size to be multiple of 64 for successful GEMM
    if v.shape[0] % 64 != 0:
        v = torch.nn.functional.pad(v, [0, 0, 0, 64 - v.shape[0] % 64])
    if mapping.is_first_pp_rank():
        tensorrt_llm_qwen.embed_tokens.vocab_embedding.weight.value = v.to(
            torch_dtype).cpu().numpy()

    # 2. lm_head
    v = [load(awq_key_list[1] + suf) for suf in awq_suffix_list]
    if v[0].shape[0] % 64 != 0:
        v[0] = torch.nn.functional.pad(v[0], [0, 0, 0, 64 - v[0].shape[0] % 64])
        scale_align = 64 * (v[0].shape[1] // group_size)
        v[1] = v[1].reshape(-1)
        v[1] = torch.nn.functional.pad(
            v[1], [0, scale_align - v[1].shape[0] % scale_align], value=1)
    if mapping.is_last_pp_rank():
        process_and_assign_weight(tensorrt_llm_qwen.lm_head, v, 1)

    # 3. ln_f
    v = load(awq_key_list[2])
    if mapping.is_last_pp_rank():
        tensorrt_llm_qwen.norm.weight.value = v.to(torch_dtype).cpu().numpy()

    # 4. Weights inside each layer
    num_hidden_layers = tensorrt_llm_qwen.num_layers
    layers_per_pipeline_stage = num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))

    for l in layers_range:
        layer_idx = l - mapping.pp_rank * layers_per_pipeline_stage
        prefix = "layers" + split_sym + str(layer_idx) + split_sym
        tensorrt_llm.logger.info(f'Process weights in layer: {layer_idx}')
        layer = tensorrt_llm_qwen.layers[layer_idx]

        # 4.1.1 attention.qkv.weight
        process_and_assign_qkv_weight(prefix + awq_key_list[3],
                                      layer.self_attn.qkv)
        # 4.1.2 attention.qkv.bias
        qkv_bias_list = []
        for x in ["q", "k", "v"]:
            x_bias = awq_qwen["model." + prefix + f"self_attn.{x}_proj.bias"].cpu()
            x_bias = torch_split(x_bias, dim=0)
            qkv_bias_list.append(x_bias)
        qkv_bias = torch.cat(qkv_bias_list, dim=0)
        layer.self_attn.qkv.bias.value = np.ascontiguousarray(qkv_bias.numpy())

        # 4.2 attention.dense
        v = [load(prefix + awq_key_list[5] + suf) for suf in awq_suffix_list]
        process_and_assign_weight(layer.self_attn.o_proj, v, 0)

        # 4.3 mlp.gate_proj
        v = [load(prefix + awq_key_list[6] + suf) for suf in awq_suffix_list]
        process_and_assign_weight(layer.mlp.gate_proj, v, 1)

        # 4.4 mlp.up_proj
        v = [load(prefix + awq_key_list[7] + suf) for suf in awq_suffix_list]
        process_and_assign_weight(layer.mlp.up_proj, v, 1)

        # 4.5 mlp.down_proj
        v = [load(prefix + awq_key_list[8] + suf) for suf in awq_suffix_list]
        process_and_assign_weight(layer.mlp.down_proj, v, 0)

        # 4.6 input_layernorm
        v = load(prefix + awq_key_list[9])
        layer.input_layernorm.weight.value = v.to(torch_dtype).cpu().numpy()

        # 4.7 post_layernorm
        v = load(prefix + awq_key_list[10])
        layer.post_attention_layernorm.weight.value = v.to(torch_dtype).cpu().numpy()

        # 4.8 attention.kv_quant_orig_scale / kv_quant_orig_scale
        if use_int8_kv_cache:
            assert ft_model_dir, "You must pass --ft_model_dir to tell TRT-LLM where to look for scales of INT8 kv cache."
            t = fromfile(
                ft_model_dir, 'model.layers.' + str(layer_idx) +
                '.attention.query_key_value.scale_y_quant_orig.bin', [1],
                np.float32)
            assert t is not None, f"{ft_model_dir} does not contain model.layers.{layer_idx}.attention.query_key_value.scale_y_quant_orig.bin"
            layer.attention.kv_orig_quant_scale.value = 1.0 / t
            layer.attention.kv_quant_orig_scale.value = t

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')