import time
from pathlib import Path

import numpy as np
import torch
import math
import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.quantization import QuantMode
from model import QWenForCausalLM


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
    for layer in range(hf_qwen.config.num_hidden_layers):
        # prefix = f'model.layers.{layer}.self_attn.'
        prefix = f'transformer.h.{layer}.attn.'
        key = f'transformer.h.{layer}.attn.c_attn.weight'
        qkv_weight = model_params[key]
        # prefix = f'transformer.h.{layer}.attn.c_proj.weight'
        # q_weight = model_params[prefix + 'q_proj.weight']
        # k_weight = model_params[prefix + 'k_proj.weight']
        # v_weight = model_params[prefix + 'v_proj.weight']
        if multi_query_mode:
            hidden_size = tensorrt_llm_qwen.hidden_size
            qkv_weight, q_weight, k_weight, v_weight = torch.split(
                qkv_weight,
                [hidden_size, hidden_size, hidden_size],
                dim=0
            )
            head_size = tensorrt_llm_qwen.hidden_size // tensorrt_llm_qwen.num_heads
            assert k_weight.shape[0] == tensor_parallel * head_size
            assert v_weight.shape[0] == tensor_parallel * head_size
            qkv_weight = [q_weight, k_weight, v_weight]
        # else:
        #     qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

        model_params[prefix + 'qkv_proj.weight'] = qkv_weight

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
    tensorrt_llm_qwen.position_embedding_cos.weight.value = torch_to_numpy(cos_weight)
    tensorrt_llm_qwen.position_embedding_sin.weight.value = torch_to_numpy(sin_weight)
    # computer logn
    logn_list = [
        math.log(i, seq_length) if i > seq_length else 1
        for i in range(1, 32768)
    ]
    logn_tensor = torch.tensor(logn_list)[None, :, None, None]
    logn_weight = torch_to_numpy(logn_tensor)
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
            elif "logn_tensor" in k:
                tensorrt_llm_qwen.layers[idx].logn_tensor.weight.value = logn_weight
            elif 'attn.qkv_proj.weight' in k:
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
                dst = tensorrt_llm_qwen.layers[idx].mlp.fc.weight
                split_v = split(v, tensor_parallel, rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_qwen.layers[
                        idx].mlp.fc.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.w2.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].mlp.gate.weight
                split_v = split(v, tensor_parallel, rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_qwen.layers[
                        idx].mlp.gate.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.c_proj.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].mlp.proj.weight
                split_v = split(v, tensor_parallel, rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_qwen.layers[
                        idx].mlp.proj.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return