import argparse
import os
import time

import tensorrt as trt
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM
from weight import load_from_hf_gpt_j

import tensorrt_llm
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.models import weight_only_quantize
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode

MODEL_NAME = "gptj"
hf_gpt = None


def get_engine_name(model, dtype, tp_size, rank):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')
    parser.add_argument(
        '--model_dir',
        type=str,
        default=None,
        help='The path to HF GPT-J model / checkpoints to read weights from')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32'])
    parser.add_argument(
        '--timing_cache',
        type=str,
        default='model.cache',
        help=
        'The path of to read timing cache from, will be ignored if the file does not exist'
    )
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--vocab_size', type=int, default=50400)
    parser.add_argument('--n_layer', type=int, default=28)
    parser.add_argument('--n_positions', type=int, default=2048)
    parser.add_argument('--n_embd', type=int, default=4096)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--hidden_act', type=str, default='gelu')
    parser.add_argument('--rotary_dim', type=int, default=64)
    parser.add_argument('--max_batch_size', type=int, default=256)
    parser.add_argument('--max_input_len', type=int, default=200)
    parser.add_argument('--max_output_len', type=int, default=200)
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument('--use_gpt_attention_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16', 'float32'])
    parser.add_argument('--use_gemm_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16', 'float32'])
    parser.add_argument('--use_weight_only_quant_matmul_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16'])
    parser.add_argument('--use_layernorm_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16', 'float32'])
    parser.add_argument('--parallel_build', default=False, action='store_true')
    parser.add_argument('--enable_context_fmha',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_context_fmha_fp32_acc',
                        default=False,
                        action='store_true')
    parser.add_argument('--gpus_per_node', type=int, default=8)
    parser.add_argument(
        '--output_dir',
        type=str,
        default='gpt_outputs',
        help=
        'The path to save the serialized engine files, timing cache file and model configs'
    )
    parser.add_argument('--remove_input_padding',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_fp8', default=False, action='store_true')
    parser.add_argument('--quantized_fp8_model_path', type=str, default=None)
    parser.add_argument(
        '--fp8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. fp8_kv_cache chooses fp8 quantization for KV'
    )
    parser.add_argument(
        '--use_inflight_batching',
        nargs='?',
        const='float16',
        default=False,
        choices=['float16', 'float32', False],
        help=
        "Activates attention plugin for inflight batching. You can specify the plugin dtype or leave blank to use the model dtype."
    )

    args = parser.parse_args()

    logger.set_level(args.log_level)

    if args.model_dir is not None:
        logger.info(f'Loading HF GPTJ model from {args.model_dir}...')
        global hf_gpt
        hf_gpt = AutoModelForCausalLM.from_pretrained(args.model_dir)
        args.n_embd = hf_gpt.config.n_embd
        args.n_head = hf_gpt.config.n_head
        args.n_layer = hf_gpt.config.n_layer
        args.n_positions = hf_gpt.config.n_positions
        args.vocab_size = hf_gpt.config.vocab_size

    args.quant_mode = QuantMode(0)
    if args.fp8_kv_cache:
        assert (
            args.use_gpt_attention_plugin or args.use_inflight_batching
        ), "You have to use GPT attention plugin or inflight batching plugin when fp8 KV cache is set"
        args.quant_mode = args.quant_mode.set_fp8_kv_cache()

    return args


def get_scaling_factors(model_path, layers=None):
    """Get the scaling factors for GPT-J model

    Returns a dictionary of scaling factors for the selected layers of the GPT-J model.

    Args:
        model_path (str): Path to the GPT-J model
        layers (list): List of layers to get the scaling factors for. If None, all layers are selected.

    Returns:
        dict: Dictionary of scaling factors for the selected layers of the GPT-J model.
        example:

        {
            'qkv_act': qkv_act_scale,
            'qkv_weights': qkv_weights_scale,
            'dense_act': dense_act_scale,
            'dense_weights': dense_weights_scale,
            'fc_act': fc_act_scale,
            'fc_weights': fc_weights_scale,
            'proj_act': proj_act_scale,
            'proj_weights': proj_weights_scale,
        }
    """
    if not os.path.exists(model_path):
        # raise RuntimeError(
        #     f"Cannot access {model_path}. Please download the model or mount the scratch path."
        # )
        logger.warning(
            f"Cannot find {model_path} to load scales of gptj. Initilize them automatically."
        )
        return {
            'fc_act': [0.99 for _ in range(args.n_layer)],
            'fc_weights': [0.99 for _ in range(args.n_layer)],
            'proj_act': [0.99 for _ in range(args.n_layer)],
            'proj_weights': [0.99 for _ in range(args.n_layer)],
            'qkv_act': [0.99 for _ in range(args.n_layer)],
            'qkv_weights': [0.99 for _ in range(args.n_layer)],
            'qkv_output':
            [5.0 for _ in range(args.n_layer)
             ],  # An experience valued observed from summarize example
            'dense_act': [0.99 for _ in range(args.n_layer)],
            'dense_weights': [0.99 for _ in range(args.n_layer)],
        }

    model = torch.load(model_path)
    n_layers = 28
    if layers is not None:
        for layer in layers:
            assert 0 >= layer and layer < n_layers, f"Layer {layer} does not exist in GPTJ model.\
                  Please enter a number between 0 and 27"

    fc_act = []
    fc_weights = []
    proj_act = []
    proj_weights = []
    qkv_act = []
    qkv_weights = []
    qkv_out = []
    dense_act = []
    dense_weights = []

    def get_qkv_out(layer):
        q_out = model[
            f"transformer.h.{layer}.attn.q_proj.output_quantizer._amax"].item()
        k_out = model[
            f"transformer.h.{layer}.attn.k_proj.output_quantizer._amax"].item()
        v_out = model[
            f"transformer.h.{layer}.attn.v_proj.output_quantizer._amax"].item()
        return max(q_out, k_out, v_out)

    def get_qkv_act(layer):
        q_act = model[
            f"transformer.h.{layer}.attn.q_proj.input_quantizer._amax"].item()
        k_act = model[
            f"transformer.h.{layer}.attn.k_proj.input_quantizer._amax"].item()
        v_act = model[
            f"transformer.h.{layer}.attn.v_proj.input_quantizer._amax"].item()
        return max(q_act, k_act, v_act)

    def get_qkv_weights(layer):
        q_weights = model[
            f"transformer.h.{layer}.attn.q_proj.weight_quantizer._amax"].item()
        k_weights = model[
            f"transformer.h.{layer}.attn.k_proj.weight_quantizer._amax"].item()
        v_weights = model[
            f"transformer.h.{layer}.attn.v_proj.weight_quantizer._amax"].item()
        return max(q_weights, k_weights, v_weights)

    if layers is None:
        layers = [x for x in range(n_layers)]
    for layer in layers:
        qkv_act.append(get_qkv_act(layer))
        qkv_weights.append(get_qkv_weights(layer))
        qkv_out.append(get_qkv_out(layer))
        dense_act.append(
            model[f"transformer.h.{layer}.attn.out_proj.input_quantizer._amax"].
            item())
        dense_weights.append(
            model[f"transformer.h.{layer}.attn.out_proj.weight_quantizer._amax"]
            .item())
        fc_act.append(
            model[f"transformer.h.{layer}.mlp.fc_in.input_quantizer._amax"].
            item())
        fc_weights.append(
            model[f"transformer.h.{layer}.mlp.fc_in.weight_quantizer._amax"].
            item())
        proj_act.append(
            model[f"transformer.h.{layer}.mlp.fc_out.input_quantizer._amax"].
            item())
        proj_weights.append(
            model[f"transformer.h.{layer}.mlp.fc_out.weight_quantizer._amax"].
            item())
    return convert_amax_to_scale(qkv_act, qkv_weights, qkv_out, dense_act,
                                 dense_weights, fc_act, fc_weights, proj_act,
                                 proj_weights)


def convert_amax_to_scale(qkv_act, qkv_weights, qkv_out, dense_act,
                          dense_weights, fc_act, fc_weights, proj_act,
                          proj_weights):
    """Convert the amax values to scaling factors for GPT-J model

    Returns a dictionary of scaling factors for the selected layers of the GPT-J model.

    Args:
        qkv_act (List[float]): List of layers' attention qkv gemm activation amax values.
        qkv_weights (List[float]): List of layers' attention qkv gemm weights amax values..
        qkv_out (List[float]): List of layers' attention qkv gemm output amax values.
        dense_act (List[float]): List of layers' attention dense gemm activation amax values.
        dense_weights (List[float]): List of layers' attention dense gemm weights amax values.
        fc_act (List[float]): List of layers' mlp fc gemm activation amax values.
        fc_weights (List[float]): List of layers' mlp fc gemm weights amax values.
        proj_act (List[float]): List of layers' mlp proj gemm activation amax values.
        proj_weights (List[float]): List of layers' mlp proj gemm weights amax values.

    Returns:
        dict: Dictionary of scaling factors for the selected layers of the GPT-J model.
    """
    scaling_factor = 448
    qkv_act_scale = [x / scaling_factor for x in qkv_act]
    qkv_weights_scale = [x / scaling_factor for x in qkv_weights]
    qkv_out_scale = [x / scaling_factor for x in qkv_out]
    dense_act_scale = [x / scaling_factor for x in dense_act]
    dense_weights_scale = [x / scaling_factor for x in dense_weights]
    fc_act_scale = [x / scaling_factor for x in fc_act]
    fc_weights_scale = [x / scaling_factor for x in fc_weights]
    proj_act_scale = [x / scaling_factor for x in proj_act]
    proj_weights_scale = [x / scaling_factor for x in proj_weights]
    gptj_scaling_factors = {
        'qkv_act': qkv_act_scale,
        'qkv_weights': qkv_weights_scale,
        'qkv_output': qkv_out_scale,
        'dense_act': dense_act_scale,
        'dense_weights': dense_weights_scale,
        'fc_act': fc_act_scale,
        'fc_weights': fc_weights_scale,
        'proj_act': proj_act_scale,
        'proj_weights': proj_weights_scale,
    }

    return gptj_scaling_factors


def build_rank_engine(builder: Builder,
                      builder_config: tensorrt_llm.builder.BuilderConfig,
                      engine_name, rank, args):
    '''
       @brief: Build the engine on the given rank.
       @param rank: The rank to build the engine.
       @param args: The cmd line arguments.
       @return: The built engine.
    '''
    kv_dtype = trt.float16 if args.dtype == 'float16' else trt.float32

    # Initialize Module
    tensorrt_llm_gpt = tensorrt_llm.models.GPTJForCausalLM(
        num_layers=args.n_layer,
        num_heads=args.n_head,
        hidden_size=args.n_embd,
        vocab_size=args.vocab_size,
        hidden_act=args.hidden_act,
        max_position_embeddings=args.n_positions,
        rotary_dim=args.rotary_dim,
        dtype=kv_dtype,
        tensor_parallel=args.world_size,  # TP only
        tensor_parallel_group=list(range(args.world_size)),  # TP only
        apply_query_key_layer_scaling=builder_config.
        apply_query_key_layer_scaling,
        fp8_mode=args.enable_fp8,
        quant_mode=args.quant_mode)
    if args.use_weight_only_quant_matmul_plugin:
        tensorrt_llm_gpt = weight_only_quantize(tensorrt_llm_gpt)

    if args.model_dir is not None:
        assert hf_gpt is not None, f'Could not load weights from hf_gpt model as it is not loaded yet.'
        if args.enable_fp8:
            gptj_scaling_factors = get_scaling_factors(
                args.quantized_fp8_model_path)
        else:
            gptj_scaling_factors = None
        load_from_hf_gpt_j(tensorrt_llm_gpt,
                           hf_gpt,
                           fp16=(args.dtype == 'float16'),
                           scaling_factors=gptj_scaling_factors)

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(
            dtype=args.use_gpt_attention_plugin)
    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
    if args.use_layernorm_plugin:
        network.plugin_config.set_layernorm_plugin(
            dtype=args.use_layernorm_plugin)
    assert not (args.enable_context_fmha and args.enable_context_fmha_fp32_acc)
    if args.enable_context_fmha:
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
    if args.enable_context_fmha_fp32_acc:
        network.plugin_config.set_context_fmha(
            ContextFMHAType.enabled_with_fp32_acc)
    if args.use_weight_only_quant_matmul_plugin:
        network.plugin_config.set_weight_only_quant_matmul_plugin(
            dtype=args.use_weight_only_quant_matmul_plugin)
    if args.world_size > 1:
        network.plugin_config.set_nccl_plugin(args.dtype)
    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()
    if args.use_inflight_batching:
        network.plugin_config.set_inflight_batching_gpt_attention_plugin(
            dtype=args.use_inflight_batching)
    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_gpt.named_parameters())

        # Forward
        inputs = tensorrt_llm_gpt.prepare_inputs(args.max_batch_size,
                                                 args.max_input_len,
                                                 args.max_output_len, True,
                                                 args.max_beam_width)
        tensorrt_llm_gpt(*inputs)

    engine = None

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    if rank == 0:
        config_path = os.path.join(args.output_dir, 'config.json')
        builder.save_config(builder_config, config_path)
    return engine


def build(rank, args):
    torch.cuda.set_device(rank % args.gpus_per_node)
    tensorrt_llm.logger.set_level(args.log_level)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # when doing serializing build, all ranks share one engine
    apply_query_key_layer_scaling = False
    builder = Builder()

    cache = None
    for cur_rank in range(args.world_size):
        # skip other ranks if parallel_build is enabled
        if args.parallel_build and cur_rank != rank:
            continue

        builder_config = builder.create_builder_config(
            name=MODEL_NAME,
            precision=args.dtype,
            timing_cache=args.timing_cache if cache is None else cache,
            tensor_parallel=args.world_size,  # TP only
            parallel_build=args.parallel_build,
            num_layers=args.n_layer,
            num_heads=args.n_head,
            hidden_size=args.n_embd,
            vocab_size=args.vocab_size,
            hidden_act=args.hidden_act,
            max_position_embeddings=args.n_positions,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            fp8=args.enable_fp8)

        engine_name = get_engine_name(MODEL_NAME, args.dtype, args.world_size,
                                      cur_rank)
        engine = build_rank_engine(builder, builder_config, engine_name,
                                   cur_rank, args)
        assert engine is not None, f'Failed to build engine for rank {cur_rank}'

        if cur_rank == 0:
            # Use in-memory timing cache for multiple builder passes.
            if not args.parallel_build:
                cache = builder_config.trt_builder_config.get_timing_cache()

        serialize_engine(engine, os.path.join(args.output_dir, engine_name))

    if rank == 0:
        ok = builder.save_timing_cache(
            builder_config, os.path.join(args.output_dir, "model.cache"))
        assert ok, "Failed to save timing cache."


if __name__ == '__main__':
    args = parse_arguments()
    tik = time.time()
    if args.parallel_build and args.world_size > 1 and \
            torch.cuda.device_count() >= args.world_size:
        logger.warning(
            f'Parallelly build TensorRT engines. Please make sure that all of the {args.world_size} GPUs are totally free.'
        )
        mp.spawn(build, nprocs=args.world_size, args=(args, ))
    else:
        args.parallel_build = False
        logger.info('Serially build TensorRT engines.')
        build(0, args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of building all {args.world_size} engines: {t}')
