import argparse
import json
import os
import time
from pathlib import Path

import tensorrt as trt
import torch
import torch.multiprocessing as mp
# from transformers import LlamaConfig, LlamaForCausalLM
# for release runing
# from transformers import (
#     AutoConfig as QWenConfig,
#     AutoModelForCausalLM as QWenForCausalLM_HF,
#     AutoTokenizer as QWenTokenizer,
# )
# for debug runing
from qwen_7b_chat.configuration_qwen import QWenConfig
from qwen_7b_chat.modeling_qwen import QWenLMHeadModel as QWenForCausalLM_HF

from model import QWenForCausalLM as QWenForCausalLM_TRT

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.models import weight_only_quantize
from tensorrt_llm.network import net_guard
from tensorrt_llm.quantization import QuantMode

from weight import load_from_hf_qwen

MODEL_NAME = "qwen"

# 2 routines: get_engine_name, serialize_engine
# are direct copy from gpt example, TODO: put in utils?

import onnx
import tensorrt as trt
from onnx import TensorProto, helper



now_dir = os.path.dirname(os.path.abspath(__file__))


def trt_dtype_to_onnx(dtype):
    if dtype == trt.float16:
        return TensorProto.DataType.FLOAT16
    elif dtype == trt.float32:
        return TensorProto.DataType.FLOAT
    elif dtype == trt.int32:
        return TensorProto.DataType.INT32
    else:
        raise TypeError("%s is not supported" % dtype)


def to_onnx(network, path):
    inputs = []
    for i in range(network.num_inputs):
        network_input = network.get_input(i)
        inputs.append(
            helper.make_tensor_value_info(
                network_input.name, trt_dtype_to_onnx(network_input.dtype),
                list(network_input.shape)))

    outputs = []
    for i in range(network.num_outputs):
        network_output = network.get_output(i)
        outputs.append(
            helper.make_tensor_value_info(
                network_output.name, trt_dtype_to_onnx(network_output.dtype),
                list(network_output.shape)))

    nodes = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        layer_inputs = []
        for j in range(layer.num_inputs):
            ipt = layer.get_input(j)
            if ipt is not None:
                layer_inputs.append(layer.get_input(j).name)
        layer_outputs = [
            layer.get_output(j).name for j in range(layer.num_outputs)
        ]
        nodes.append(
            helper.make_node(str(layer.type),
                             name=layer.name,
                             inputs=layer_inputs,
                             outputs=layer_outputs,
                             domain="com.nvidia"))

    onnx_model = helper.make_model(helper.make_graph(nodes,
                                                     'attention',
                                                     inputs,
                                                     outputs,
                                                     initializer=None),
                                   producer_name='NVIDIA')
    onnx.save(onnx_model, path)


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
    parser.add_argument('--model_dir', type=str, default=os.path.join(now_dir, "qwen_7b_chat"))
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument(
        '--timing_cache',
        type=str,
        default='model.cache',
        help=
        'The path of to read timing cache from, will be ignored if the file does not exist'
    )
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--n_layer', type=int, default=32)
    parser.add_argument('--n_positions', type=int, default=2048)
    parser.add_argument('--n_embd', type=int, default=4096)
    parser.add_argument('--n_head', type=int, default=32)
    parser.add_argument('--n_kv_head', type=int, default=None)
    parser.add_argument('--multiple_of', type=int, default=None)
    parser.add_argument('--ffn_dim_multiplier', type=int, default=1)
    parser.add_argument('--inter_size', type=int, default=11008)
    parser.add_argument('--hidden_act', type=str, default='silu')
    parser.add_argument('--max_batch_size', type=int, default=8)
    parser.add_argument('--max_input_len', type=int, default=2048)
    parser.add_argument('--max_output_len', type=int, default=512)
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument('--use_gpt_attention_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default="float16",
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--use_gemm_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default="float16",
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--parallel_build', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--enable_debug_output',
                        default=False,
                        action='store_true')
    parser.add_argument('--gpus_per_node', type=int, default=8)
    parser.add_argument('--builder_opt', type=int, default=None)
    parser.add_argument(
        '--output_dir',
        type=str,
        default='trt_engines',
        help=
        'The path to save the serialized engine files, timing cache file and model configs'
    )
    parser.add_argument('--remove_input_padding',
                        default=False,
                        action='store_true')

    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision')

    parser.add_argument(
        '--weight_only_precision',
        const='int8',
        type=str,
        nargs='?',
        default='int8',
        choices=['int8', 'int4'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )

    args = parser.parse_args()

    if args.use_weight_only:
        args.quant_mode = QuantMode.use_weight_only(
            args.weight_only_precision == 'int4')
    else:
        args.quant_mode = QuantMode(0)
    # Since gpt_attenttion_plugin is the only way to apply RoPE now,
    # force use the plugin for now with the correct data type.
    args.use_gpt_attention_plugin = args.dtype
    if args.model_dir is not None:
        hf_config = QWenConfig.from_pretrained(args.model_dir)
        args.inter_size = hf_config.intermediate_size  # override the inter_size for QWen
        args.n_embd = hf_config.hidden_size
        args.n_head = hf_config.num_attention_heads
        if hasattr(hf_config, "num_key_value_heads"):
            args.n_kv_head = hf_config.num_key_value_heads
        args.n_layer = hf_config.num_hidden_layers
        args.n_positions = hf_config.max_position_embeddings
        args.vocab_size = hf_config.vocab_size
        args.hidden_act = "silu"
    assert args.use_gpt_attention_plugin is not None, "QWen must use gpt attention plugin"
    if args.n_kv_head is not None and args.n_kv_head != args.n_head:
        assert args.n_kv_head == args.world_size, \
        "The current implementation of GQA requires the number of K/V heads to match the number of GPUs." \
        "This limitation will be removed in a future version."

    return args


def build_rank_engine(builder: Builder,
                      builder_config: tensorrt_llm.builder.BuilderConfig,
                      engine_name, rank, multi_query_mode, args):
    '''
       @brief: Build the engine on the given rank.
       @param rank: The rank to build the engine.
       @param args: The cmd line arguments.
       @return: The built engine.
    '''
    kv_dtype = str_dtype_to_trt(args.dtype)

    # Initialize Module
    tensorrt_llm_qwen = QWenForCausalLM_TRT(
        num_layers=args.n_layer,
        num_heads=args.n_head,
        hidden_size=args.n_embd,
        vocab_size=args.vocab_size,
        hidden_act=args.hidden_act,
        max_position_embeddings=args.n_positions,
        dtype=kv_dtype,
        mlp_hidden_size=args.inter_size,
        neox_rotary_style=True,
        multi_query_mode=multi_query_mode,
        tensor_parallel=args.world_size,  # TP only
        tensor_parallel_group=list(range(args.world_size)))
    if args.use_weight_only and args.weight_only_precision == 'int8':
        tensorrt_llm_qwen = weight_only_quantize(
            tensorrt_llm_qwen,
            QuantMode.use_weight_only()
        )
    elif args.use_weight_only and args.weight_only_precision == 'int4':
        tensorrt_llm_qwen = weight_only_quantize(
            tensorrt_llm_qwen,
            QuantMode.use_weight_only(use_int4_weights=True)
        )
    elif args.model_dir is not None:
        logger.info(f'Loading HF QWen ... from {args.model_dir}')
        tik = time.time()
        hf_qwen = QWenForCausalLM_HF.from_pretrained(
            args.model_dir,
            device_map={
                "transformer": "cpu",
                "lm_head": "cpu"
            },  # Load to CPU memory
            torch_dtype="auto"
        )
        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        logger.info(f'HF QWen loaded. Total time: {t}')
        load_from_hf_qwen(
            tensorrt_llm_qwen,
            hf_qwen,
            rank,
            args.world_size,
            dtype=args.dtype,
            multi_query_mode=multi_query_mode
        )
        del hf_qwen

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(
            dtype=args.use_gpt_attention_plugin)
    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
    if args.use_weight_only:
        network.plugin_config.set_weight_only_quant_matmul_plugin(
            dtype='float16')
    if args.world_size > 1:
        network.plugin_config.set_nccl_plugin(args.dtype)
    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()

    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_qwen.named_parameters())

        # Forward
        inputs = tensorrt_llm_qwen.prepare_inputs(
            args.max_batch_size,
            args.max_input_len,
            args.max_output_len,
            True,
            args.max_beam_width
        )
        tensorrt_llm_qwen(*inputs)
        if args.enable_debug_output:
            # mark intermediate nodes' outputs
            for k, v in tensorrt_llm_qwen.named_network_outputs():
                v = v.trt_tensor
                v.name = k
                network.trt_network.mark_output(v)
                v.dtype = kv_dtype
        if args.visualize:
            model_path = os.path.join(args.output_dir, 'test.onnx')
            to_onnx(network.trt_network, model_path)

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
    multi_query_mode = (args.n_kv_head
                        is not None) and (args.n_kv_head != args.n_head)

    # when doing serializing build, all ranks share one engine
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
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            int8=args.quant_mode.has_act_and_weight_quant(),
            opt_level=args.builder_opt,
            multi_query_mode=multi_query_mode)
        engine_name = get_engine_name(MODEL_NAME, args.dtype, args.world_size,
                                      cur_rank)
        engine = build_rank_engine(builder, builder_config, engine_name,
                                   cur_rank, multi_query_mode, args)
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
    logger.set_level(args.log_level)
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
