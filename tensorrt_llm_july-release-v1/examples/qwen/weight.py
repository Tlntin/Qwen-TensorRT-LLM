import time
import configparser
from pathlib import Path
import os
from tqdm import trange
import numpy as np
import torch
import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_np, pad_vocab_size, torch_to_numpy
from tensorrt_llm.quantization import QuantMode
from model import QWenForCausalLM

def gen_suffix(rank, use_smooth_quant, quant_per_channel):
    suffix = f"{rank}.bin"
    if use_smooth_quant:
        sq_prefix = "int8."
        if quant_per_channel:
            sq_prefix += "col."
        suffix = sq_prefix + suffix
    return suffix

def extract_layer_idx(name):
    ss = name.split('.')
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

    vocab_size = qwen_config.getint('qwen', 'vocab_size')
    hidden_size = qwen_config.getint('qwen', 'hidden_size')
    inter_size = qwen_config.getint('qwen', 'intermediate_size', fallback=None)
    num_hidden_layers = qwen_config.getint(
        "qwen",
        "num_hidden_layers",
        fallback=32,
    )
    max_position_embeddings = qwen_config.getint(
        "qwen", "max_position_embeddings", fallback=8192)
    kv_channels = qwen_config.getint('qwen', 'kv_channels', fallback=128)
    rotary_pct = qwen_config.getfloat('qwen', 'rotary_pct', fallback=0.0)
    rotary_emb_base = qwen_config.getint(
        'qwen', 'rotary_emb_base', fallback=10000
    )
    multi_query_mode = qwen_config.getboolean(
        'qwen',
        'multi_query_mode',
        fallback=False
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
        max_position_embeddings
    )

def load_from_ft(tensorrt_llm_qwen: QWenForCausalLM,
                 dir_path,
                 rank=0,
                 tensor_parallel=1,
                 dtype='float16',
                 share_embedding_table=False,
                 parallel_embedding_table=False,
                 multi_query_mode=False):
    tensorrt_llm.logger.info('Loading weights from FT...')
    tik = time.time()
    quant_mode = getattr(tensorrt_llm_qwen, 'quant_mode', QuantMode(0))
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
        max_position_embeddings
    ) = parse_ft_config(
        Path(dir_path) / 'config.ini')
    np_dtype = str_dtype_to_np(dtype)

    def fromfile(dir_path, name, shape=None, dtype=np.float16):
        dtype = np_dtype if dtype is None else dtype
        p = dir_path + '/' + name
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
            t = fromfile(dir_path, f"{basename}scale_w_quant_orig.{suffix}",
                         col_shape, np.float32)
            module.per_channel_scale.value = t
        else:
            t = fromfile(dir_path, f"{basename}scale_x_orig_quant.bin", [1],
                         np.float32)
            pre_scale_weight.value = t
            t = fromfile(dir_path, f"{basename}scale_y_accum_quant.{suffix}",
                         col_shape, np.float32)
            module.per_channel_scale.value = t
            t = fromfile(dir_path, f"{basename}scale_y_quant_orig.bin", [1, 1],
                         np.float32)
            module.act_scale.value = t

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

    def sq_trick(x):
        return x.view(np.float32) if use_smooth_quant else x

    # Debug
    suffix = gen_suffix(rank, use_smooth_quant, quant_per_channel)
    # The type of weights.
    w_type = np_dtype if not use_smooth_quant else np.int8

    # pe = fromfile(dir_path, 'model.wpe.bin', [n_positions, n_embd])
    # if pe is not None:
    #     tensorrt_llm_qwen.embedding.position_embedding.weight.value = (pe)

    # set for rope embedding
    inv_freq = 1.0 / (rotary_emb_base ** (
        torch.arange(0, kv_channels, 2).float() / kv_channels)
    )
    value_table = torch.matmul(
        torch.arange(max_position_embeddings).float().reshape(-1, 1),
        torch.concat([inv_freq, inv_freq], dim=0).reshape(1, -1)
    ).reshape(max_position_embeddings, len(inv_freq) * 2)
    cos_weight = torch.cos(value_table).float()
    sin_weight = torch.sin(value_table).float()
    tensorrt_llm_qwen.rope.position_embedding_cos.weight.value = torch_to_numpy(cos_weight)
    tensorrt_llm_qwen.rope.position_embedding_sin.weight.value = torch_to_numpy(sin_weight)

    # breakpoint()
    vocab_embedding_weight = fromfile(dir_path, 'vocab_embedding.weight.bin',
                                      [vocab_size, hidden_size])
    tensorrt_llm_qwen.vocab_embedding.weight.value = vocab_embedding_weight


    lm_head_weight = fromfile(dir_path, 'lm_head.weight.bin',
                                  [vocab_size, hidden_size])
    tensorrt_llm_qwen.lm_head.weight.value = np.ascontiguousarray(
            split(lm_head_weight, tensor_parallel, rank))

    tensorrt_llm_qwen.ln_f.weight.value = fromfile(dir_path, 'ln_f.weight.bin')

    for i in trange(num_hidden_layers, desc="load weights"):
        c_attn_out_dim = (3 * hidden_size //
                          tensor_parallel) if not multi_query_mode else (
                              hidden_size // tensor_parallel +
                              (hidden_size // num_hidden_layers) * 2)

        tensorrt_llm_qwen.layers[i].ln_1.weight.value = fromfile(
            dir_path, 'model.layers.' + str(i) + '.ln_1.weight.bin'
        )

        dst = tensorrt_llm_qwen.layers[i].ln_2.weight
        dst.value = fromfile(
            dir_path, 'model.layers.' + str(i) + '.ln_2.weight.bin')

        t = fromfile(
            dir_path, 'model.layers.' + str(i) + '.attention.qkv.weight.' + suffix,
            [hidden_size, c_attn_out_dim],
            w_type
        )
        #breakpoint()
        if t is not None:
            dst = tensorrt_llm_qwen.layers[i].attention.qkv.weight
            if use_smooth_quant:
                dst.value = sq_trick(
                    np.ascontiguousarray(np.transpose(t, [1, 0])))
                set_smoothquant_scale_factors(
                    tensorrt_llm_qwen.layers[i].attention.qkv,
                    tensorrt_llm_qwen.layers[i].ln_1.scale_to_int,
                    dir_path,
                    'model.layers.' + str(i) + '.attention.qkv.',
                    [1, c_attn_out_dim],
                    quant_per_token_dyn,
                    quant_per_channel,
                    rank=rank,
                    is_qkv=True)
            elif use_weight_only:
                # t = np.ascontiguousarray(np.transpose(t, [1, 0]))
                processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(t), plugin_weight_only_quant_type)
                # workaround for trt not supporting int8 inputs in plugins currently
                dst.value = processed_torch_weights.view(
                    dtype=torch.float32).numpy()
                scales = tensorrt_llm_qwen.layers[i].attention.qkv.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        dst = tensorrt_llm_qwen.layers[i].attention.qkv.bias
        t = fromfile(
            dir_path, 'model.layers.' + str(i) +
            '.attention.qkv.bias.' + str(rank) + '.bin', [c_attn_out_dim])
        dst.value = np.ascontiguousarray(t)

        dst = tensorrt_llm_qwen.layers[i].attention.dense.weight
        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.attention.dense.weight.' + suffix,
            [hidden_size // tensor_parallel, hidden_size], w_type)
        if use_smooth_quant:
            dst.value = sq_trick(np.ascontiguousarray(np.transpose(t, [1, 0])))
            dense_scale = getattr(tensorrt_llm_qwen.layers[i].attention,
                                  "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_qwen.layers[i].attention.dense, dense_scale,
                dir_path, 'model.layers.' + str(i) + '.attention.dense.',
                [1, hidden_size],
                quant_per_token_dyn,
                quant_per_channel,
            )
            
        elif use_weight_only:
            # t = np.ascontiguousarray(np.transpose(t, [1, 0]))
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            # workaround for trt not supporting int8 inputs in plugins currently
            dst.value = processed_torch_weights.view(
                dtype=torch.float32).numpy()
            scales = tensorrt_llm_qwen.layers[i].attention.dense.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.mlp.w1.weight.' + suffix,
            [hidden_size, inter_size // tensor_parallel // 2],
            w_type
        )
        if use_smooth_quant:
            tensorrt_llm_qwen.layers[i].mlp.w1.weight.value = sq_trick(
                np.ascontiguousarray(np.transpose(t, [1, 0])))
            set_smoothquant_scale_factors(
                tensorrt_llm_qwen.layers[i].mlp.w1,
                tensorrt_llm_qwen.layers[i].ln_2.scale_to_int,
                dir_path,
                'model.layers.' + str(i) + '.mlp.w1.',
                [1, inter_size // tensor_parallel//2],
                quant_per_token_dyn,
                quant_per_channel,
                rank=rank
            )
        elif use_weight_only:
            dst = tensorrt_llm_qwen.layers[i].mlp.w1.weight
            # t = np.ascontiguousarray(np.transpose(t, [1, 0]))
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            # workaround for trt not supporting int8 inputs in plugins currently
            dst.value = processed_torch_weights.view(
                dtype=torch.float32).numpy()
            scales = tensorrt_llm_qwen.layers[i].mlp.w1.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_qwen.layers[i].mlp.w1.weight.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.mlp.w2.weight.' + suffix,
            [hidden_size, inter_size // tensor_parallel//2], w_type)
        if use_smooth_quant:
            tensorrt_llm_qwen.layers[i].mlp.w2.weight.value = sq_trick(
                np.ascontiguousarray(np.transpose(t, [1, 0])))
            set_smoothquant_scale_factors(
                tensorrt_llm_qwen.layers[i].mlp.w2,
                tensorrt_llm_qwen.layers[i].ln_2.scale_to_int,
                dir_path,
                'model.layers.' + str(i) + '.mlp.w2.',
                [1, inter_size // tensor_parallel//2],
                quant_per_token_dyn,
                quant_per_channel,
                rank=rank
            )
        elif use_weight_only:
            dst = tensorrt_llm_qwen.layers[i].mlp.w2.weight
            # t = np.ascontiguousarray(np.transpose(t, [1, 0]))
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            # workaround for trt not supporting int8 inputs in plugins currently
            dst.value = processed_torch_weights.view(
                dtype=torch.float32).numpy()
            scales = tensorrt_llm_qwen.layers[i].mlp.w2.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_qwen.layers[i].mlp.w2.weight.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.mlp.c_proj.weight.' + suffix,
            [inter_size // tensor_parallel//2, hidden_size],
            w_type
        )
        if use_smooth_quant:
            tensorrt_llm_qwen.layers[i].mlp.c_proj.weight.value = sq_trick(
                np.ascontiguousarray(np.transpose(t, [1, 0])))
            proj_scale = getattr(tensorrt_llm_qwen.layers[i].mlp,
                                 "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_qwen.layers[i].mlp.c_proj, proj_scale, dir_path,
                'model.layers.' + str(i) + '.mlp.c_proj.', [1, hidden_size],
                quant_per_token_dyn, quant_per_channel)
        elif use_weight_only:
            dst = tensorrt_llm_qwen.layers[i].mlp.c_proj.weight
            # t = np.ascontiguousarray(np.transpose(t, [1, 0]))
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            # workaround for trt not supporting int8 inputs in plugins currently
            dst.value = processed_torch_weights.view(
                dtype=torch.float32).numpy()
            scales = tensorrt_llm_qwen.layers[i].mlp.c_proj.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_qwen.layers[i].mlp.c_proj.weight.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        if use_int8_kv_cache:
            t = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.attention.query_key_value.scale_y_quant_orig.bin', [1],
                np.float32)
            tensorrt_llm_qwen.layers[
                i].attention.kv_orig_quant_scale.value = 1.0 / t
            tensorrt_llm_qwen.layers[i].attention.kv_quant_orig_scale.value = t

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')


def load_from_hf_qwen(tensorrt_llm_qwen: QWenForCausalLM,
                       hf_qwen,
                       rank=0,
                       tensor_parallel=1,
                       seq_length=2048,
                       max_position_embeddings=8192,
                       rotary_emb_base=10000,
                       kv_channels=128,
                       dtype="float32",
                       multi_query_mode=False):
    tensorrt_llm.logger.info('Loading weights from HF QWen...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_qwen, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    model_params = dict(hf_qwen.named_parameters())
    # for layer in range(hf_qwen.config.num_hidden_layers):
    #     # prefix = f'model.layers.{layer}.self_attn.'
    #     prefix = f'transformer.h.{layer}.attn.'
    #     key = f'transformer.h.{layer}.attn.c_attn.weight'
    #     qkv_weight = model_params[key]
    #     # prefix = f'transformer.h.{layer}.attn.c_proj.weight'
    #     # q_weight = model_params[prefix + 'q_proj.weight']
    #     # k_weight = model_params[prefix + 'k_proj.weight']
    #     # v_weight = model_params[prefix + 'v_proj.weight']
    #     if multi_query_mode:
    #         hidden_size = tensorrt_llm_qwen.hidden_size
    #         qkv_weight, q_weight, k_weight, v_weight = torch.split(
    #             qkv_weight,
    #             [hidden_size, hidden_size, hidden_size],
    #             dim=0
    #         )
    #         head_size = tensorrt_llm_qwen.hidden_size // tensorrt_llm_qwen.num_heads
    #         assert k_weight.shape[0] == tensor_parallel * head_size
    #         assert v_weight.shape[0] == tensor_parallel * head_size
    #         qkv_weight = [q_weight, k_weight, v_weight]
    #     # else:
    #     #     qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

    #     model_params[prefix + 'qkv_proj.weight'] = qkv_weight

    torch_dtype = str_dtype_to_torch(dtype)
    # set for rope embedding
    inv_freq = 1.0 / (rotary_emb_base ** (
        torch.arange(0, kv_channels, 2).float() / kv_channels)
    )
    value_table = torch.matmul(
        torch.arange(max_position_embeddings).float().reshape(-1, 1),
        torch.concat([inv_freq, inv_freq], dim=0).reshape(1, -1)
    ).reshape(max_position_embeddings, len(inv_freq) * 2)
    cos_weight = torch.cos(value_table).float()
    sin_weight = torch.sin(value_table).float()
    tensorrt_llm_qwen.rope.position_embedding_cos.weight.value = torch_to_numpy(cos_weight)
    tensorrt_llm_qwen.rope.position_embedding_sin.weight.value = torch_to_numpy(sin_weight)
    for k, v in model_params.items():
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
        if 'transformer.wte.weight' in k:
            tensorrt_llm_qwen.vocab_embedding.weight.value = v
        elif 'transformer.ln_f.weight' in k:
            tensorrt_llm_qwen.ln_f.weight.value = v
        elif 'lm_head.weight' in k:
            tensorrt_llm_qwen.lm_head.weight.value = np.ascontiguousarray(
                split(v, tensor_parallel, rank))
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if idx >= tensorrt_llm_qwen.num_layers:
                continue
            if 'ln_1.weight' in k:
                tensorrt_llm_qwen.layers[idx].ln_1.weight.value = v
            elif 'ln_2.weight' in k:
                tensorrt_llm_qwen.layers[idx].ln_2.weight.value = v
            elif 'attn.c_attn.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].attention.qkv.weight
                if multi_query_mode:
                    assert isinstance(v, list) and len(v) == 3
                    wq = split(v[0], tensor_parallel, rank)
                    wk = split(v[1], tensor_parallel, rank)
                    wv = split(v[2], tensor_parallel, rank)
                    split_v = np.concatenate((wq, wk, wv))
                else:
                    q_emb = v.shape[0] // 3
                    model_emb = v.shape[1]
                    v = v.reshape(3, q_emb, model_emb)
                    split_v = split(v, tensor_parallel, rank, dim=1)
                    split_v = split_v.reshape(3 * (q_emb // tensor_parallel),
                                              model_emb)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_qwen.layers[
                        idx].attention.qkv.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'attn.c_attn.bias' in k:
                dst = tensorrt_llm_qwen.layers[idx].attention.qkv.bias
                if multi_query_mode:
                    assert isinstance(v, list) and len(v) == 3
                    wq = split(v[0], tensor_parallel, rank)
                    wk = split(v[1], tensor_parallel, rank)
                    wv = split(v[2], tensor_parallel, rank)
                    split_v = np.concatenate((wq, wk, wv))
                else:
                    q_emb = v.shape[0] // 3
                    v = v.reshape(3, q_emb)
                    split_v = split(v, tensor_parallel, rank, dim=1)
                    split_v = split_v.reshape(3 * (q_emb // tensor_parallel))
                dst.value = np.ascontiguousarray(split_v)
            elif 'attn.c_proj.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].attention.dense.weight
                split_v = split(v, tensor_parallel, rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_qwen.layers[
                        idx].attention.dense.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.w1.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].mlp.w1.weight
                split_v = split(v, tensor_parallel, rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_qwen.layers[
                        idx].mlp.w1.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.w2.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].mlp.w2.weight
                split_v = split(v, tensor_parallel, rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_qwen.layers[
                        idx].mlp.w2.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.c_proj.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].mlp.c_proj.weight
                split_v = split(v, tensor_parallel, rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_qwen.layers[
                        idx].mlp.c_proj.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            else:
                print("unknow key: ", k)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return