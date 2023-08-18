import time
from pathlib import Path

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.quantization import QuantMode


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


def load_from_hf_llama(tensorrt_llm_llama,
                       hf_llama,
                       rank=0,
                       tensor_parallel=1,
                       dtype="float32",
                       multi_query_mode=False):
    tensorrt_llm.logger.info('Loading weights from HF LLaMA...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_llama, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    model_params = dict(hf_llama.named_parameters())
    for l in range(hf_llama.config.num_hidden_layers):
        prefix = f'model.layers.{l}.self_attn.'
        q_weight = model_params[prefix + 'q_proj.weight']
        k_weight = model_params[prefix + 'k_proj.weight']
        v_weight = model_params[prefix + 'v_proj.weight']
        if multi_query_mode:
            head_size = tensorrt_llm_llama.hidden_size // tensorrt_llm_llama.num_heads
            assert k_weight.shape[0] == tensor_parallel * head_size
            assert v_weight.shape[0] == tensor_parallel * head_size
            qkv_weight = [q_weight, k_weight, v_weight]
        else:
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

        model_params[prefix + 'qkv_proj.weight'] = qkv_weight

    torch_dtype = str_dtype_to_torch(dtype)
    for k, v in model_params.items():
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
        if 'model.embed_tokens.weight' in k:
            tensorrt_llm_llama.vocab_embedding.weight.value = v
        elif 'model.norm.weight' in k:
            tensorrt_llm_llama.ln_f.weight.value = v
        elif 'lm_head.weight' in k:
            tensorrt_llm_llama.lm_head.weight.value = np.ascontiguousarray(
                split(v, tensor_parallel, rank))
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if idx >= tensorrt_llm_llama.num_layers:
                continue
            if 'input_layernorm.weight' in k:
                tensorrt_llm_llama.layers[idx].input_layernorm.weight.value = v
            elif 'post_attention_layernorm.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].post_layernorm.weight
                dst.value = v
            elif 'self_attn.qkv_proj.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].attention.qkv.weight
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
                    scales = tensorrt_llm_llama.layers[
                        idx].attention.qkv.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'self_attn.o_proj.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].attention.dense.weight
                split_v = split(v, tensor_parallel, rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_llama.layers[
                        idx].attention.dense.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.up_proj.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].mlp.gate.weight
                split_v = split(v, tensor_parallel, rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_llama.layers[
                        idx].mlp.gate.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.down_proj.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].mlp.proj.weight
                split_v = split(v, tensor_parallel, rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_llama.layers[
                        idx].mlp.proj.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.gate_proj.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].mlp.fc.weight
                split_v = split(v, tensor_parallel, rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_llama.layers[
                        idx].mlp.fc.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return


def load_from_meta_llama(
        tensorrt_llm_llama: tensorrt_llm.models.LLaMAForCausalLM,
        meta_ckpt_dir,
        rank=0,
        tensor_parallel=1,
        dtype="float32",
        multi_query_mode=False):

    def permute(w, nH, d, dH):
        # due to MQA's wk, nH*dH != d could be true
        return w.view(nH, dH // 2, 2, d).transpose(1, 2).reshape(nH * dH, d)

    if not hasattr(load_from_meta_llama, "saved_embed"):
        load_from_meta_llama.saved_embed = None

    def gather_embedding(cur_embed, name: str):
        if tensor_parallel == 1:
            return cur_embed
        if load_from_meta_llama.saved_embed is None:
            embeds = [None] * tensor_parallel
            embeds[rank] = torch.tensor(cur_embed)
            for i in range(tensor_parallel):
                if i != rank:
                    ckpt = torch.load(Path(meta_ckpt_dir,
                                           f"consolidated.{i:02d}.pth"),
                                      map_location="cpu")
                    embeds[i] = ckpt[name]
            embed = torch.cat(embeds, dim=1)
            load_from_meta_llama.saved_embed = embed.numpy(
            )  # cache the embedding, not needed if no refit
        return load_from_meta_llama.saved_embed

    tensorrt_llm.logger.info('Loading weights from Meta LLaMA checkpoints ...')
    tik = time.time()
    ckpts = list(Path(meta_ckpt_dir).glob("consolidated.*.pth"))
    num_ckpts = len(ckpts)
    assert num_ckpts == tensor_parallel, f"TP={tensor_parallel} must be equal to the number of checkpoint files ({num_ckpts}) to use this loader."
    # NOTE:
    #   If multi_query_mode is enabled, this function
    #   assumes there are `tensor_parallel` number of shards
    #   and each shard contains only 1 KV Head (for now).
    if multi_query_mode:
        num_kv_heads = tensor_parallel
    else:
        num_kv_heads = tensorrt_llm_llama.num_heads
    head_size = tensorrt_llm_llama.hidden_size // tensorrt_llm_llama.num_heads
    ckpt = torch.load(Path(meta_ckpt_dir, f"consolidated.{rank:02d}.pth"),
                      map_location="cpu")
    for l in range(tensorrt_llm_llama.num_layers):
        prefix = f'layers.{l}.attention.'
        q_weight = permute(ckpt[prefix + 'wq.weight'],
                           nH=(tensorrt_llm_llama.num_heads // tensor_parallel),
                           d=tensorrt_llm_llama.hidden_size,
                           dH=head_size)
        k_weight = permute(ckpt[prefix + 'wk.weight'],
                           nH=(num_kv_heads // tensor_parallel),
                           d=tensorrt_llm_llama.hidden_size,
                           dH=head_size)
        v_weight = ckpt[prefix + 'wv.weight']

        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        ckpt[prefix + 'qkv.weight'] = qkv_weight

    torch_dtype = str_dtype_to_torch(dtype)
    for k, v in ckpt.items():
        v = v.to(torch_dtype).detach().cpu().numpy()
        if "tok_embeddings" in k:
            v = gather_embedding(
                v, k)  # TODO: Won't be needed once Embedding layer supports TP
            tensorrt_llm_llama.vocab_embedding.weight.value = v
        elif "output" in k:
            tensorrt_llm_llama.lm_head.weight.value = v
        elif k == "norm.weight":
            tensorrt_llm_llama.ln_f.weight.value = v
        else:
            # layer specific weights
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if idx >= tensorrt_llm_llama.num_layers:
                continue
            if 'attention_norm.weight' in k:
                tensorrt_llm_llama.layers[idx].input_layernorm.weight.value = v
            elif 'ffn_norm.weight' in k:
                tensorrt_llm_llama.layers[idx].post_layernorm.weight.value = v
            elif 'feed_forward.w3.weight' in k:
                tensorrt_llm_llama.layers[idx].mlp.gate.weight.value = v
            elif 'feed_forward.w2.weight' in k:
                tensorrt_llm_llama.layers[idx].mlp.proj.weight.value = v
            elif 'feed_forward.w1.weight' in k:
                tensorrt_llm_llama.layers[idx].mlp.fc.weight.value = v
            elif 'attention.wo.weight' in k:
                tensorrt_llm_llama.layers[idx].attention.dense.weight.value = v
            elif 'attention.qkv.weight' in k:
                tensorrt_llm_llama.layers[idx].attention.qkv.weight.value = v

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return
