import time

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.models import ChatGLM2HeadModel
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.mapping import Mapping
from model import ChatGLM2HeadModel, ChatGLM2Model


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

def dup_kv_weight(v, kv_num_head, num_head):
    reps = num_head // kv_num_head
    head_size = v.shape[0] // kv_num_head
    v = torch.from_numpy(v)
    v = v.reshape(kv_num_head, head_size,
                  -1)[:, None, :, :].expand(kv_num_head, reps, head_size, v.shape[1])
    return v.reshape(kv_num_head * reps * head_size, -1).clone()

def dup_kv_bias(v, kv_num_head, num_head):
    reps = num_head // kv_num_head
    head_size = v.shape[0] // kv_num_head
    v = torch.from_numpy(v)
    v = v.reshape(kv_num_head, head_size)[:, None, :].expand(kv_num_head, reps, head_size)
    return v.reshape(kv_num_head * reps * head_size).clone()

def qkv_split(v, tp_size, hidden_size, kv_num_head, num_head, kv_channels, idx, bias=False, dim=0):
    q = v[:hidden_size]
    k_start = hidden_size
    k_end = hidden_size+kv_num_head*kv_channels
    k = v[k_start:k_end]
    v_start = k_end
    v_end = v_start + kv_num_head*kv_channels
    v_w = v[v_start:v_end]

    if bias:
        k_dup = dup_kv_bias(k, kv_num_head, num_head)
        v_dup = dup_kv_bias(v_w, kv_num_head, num_head)
    else:
        k_dup = dup_kv_weight(k, kv_num_head, num_head)
        v_dup = dup_kv_weight(v_w, kv_num_head, num_head)

    q_tmp = np.ascontiguousarray(np.split(q, tp_size)[idx])
    k_tmp = np.ascontiguousarray(np.split(k_dup, tp_size)[idx])
    v_tmp = np.ascontiguousarray(np.split(v_dup, tp_size)[idx])
    return np.concatenate([q_tmp, k_tmp, v_tmp], 0)


def load_from_hf_chatglm2_6b(tensorrt_llm_model,
                           hf_chatglm6b,
                           mapping=Mapping(),
                           #rank=0,
                           #tensor_parallel=1,
                           fp16=False):
    tensorrt_llm.logger.info('Loading weights from HF GPT...')
    tik = time.time()

    # Determine the quantization mode.
    quant_mode = getattr(tensorrt_llm_model, "quant_mode", QuantMode(0))

    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
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
    #for l in range(tensorrt_llm_model.num_layers):

    model_params = dict(hf_chatglm6b.named_parameters())
    kv_num_head = tensorrt_llm_model.multi_query_group_num
    kv_channels = tensorrt_llm_model.kv_channels
    vocab_size = tensorrt_llm_model._vocab_size
    hidden_size = tensorrt_llm_model._hidden_size
    num_head = tensorrt_llm_model._num_heads

    model_params = dict(hf_chatglm6b.named_parameters())

    #breakpoint()
    for k, v in model_params.items():
        torch_dtype = torch.float16 if fp16 else torch.float32
        #v = v.to(torch_dtype).cpu().numpy()
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
        #breakpoint()
        if 'word_embeddings.weight' in k:
            tensorrt_llm_model.embedding.weight.value = v
        #elif 'attention.rotary_emb.inv_freq' in k:
        #    nMaxSL = 2048
        #    valueTable = np.matmul(np.arange(nMaxSL, dtype=np.float32).reshape(-1, 1),
        #                           np.concatenate([v, v],
        #                           axis=0).reshape(1, -1)).reshape(nMaxSL,
        #                           len(v) * 2)
        #    tensorrt_llm_model.position_embedding_cos.weight.value = np.cos(valueTable).astype(np.float16)
        #    tensorrt_llm_model.position_embedding_sin.weight.value = np.sin(valueTable).astype(np.float16)
        elif 'final_layernorm.weight' in k:
            tensorrt_llm_model.encoder.final_layernorm.weight.value = v
        elif 'output_layer.weight' in k:
            if vocab_size % mapping.tp_size != 0:
            # padding
                vocab_size_padded = tensorrt_llm_model.lm_head.out_features * mapping.tp_size
                pad_width = vocab_size_padded - vocab_size
                lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)),
                                'constant',
                                constant_values=0)
            tensorrt_llm_model.lm_head.weight.value = np.ascontiguousarray(
                split(v, mapping.tp_size, mapping.rank))
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if 'input_layernorm.weight' in k:
                tensorrt_llm_model.encoder.layers[idx].input_layernorm.weight.value = v
            elif 'attention.query_key_value.weight' in k:
                # HF-GPT uses Conv1D instead of Linear
                #v = v.transpose()
                dst = tensorrt_llm_model.encoder.layers[idx].self_attention.qkv.weight
                split_v = qkv_split(v, mapping.tp_size, hidden_size, kv_num_head, num_head, kv_channels, mapping.rank, False, dim = 0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_model.encoder.layers[
                        idx].self_attention.qkv.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'attention.query_key_value.bias' in k:
                dst = tensorrt_llm_model.encoder.layers[idx].self_attention.qkv.bias
                dst.value = np.ascontiguousarray(qkv_split(v, mapping.tp_size, hidden_size, kv_num_head, num_head, kv_channels, mapping.rank, True))
            elif 'attention.dense.weight' in k:
                #v = v.transpose()
                dst = tensorrt_llm_model.encoder.layers[idx].self_attention.dense.weight
                split_v = split(v, mapping.tp_size, mapping.rank, dim = 1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_model.encoder.layers[
                        idx].self_attention.dense.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'post_attention_layernorm.weight' in k:
                dst = tensorrt_llm_model.encoder.layers[idx].post_attention_layernorm.weight
                dst.value = v
            elif 'mlp.dense_h_to_4h.weight' in k:
                h_to_4h_weight = torch.split(torch.from_numpy(v), v.shape[0] // 2, 0)
                v = torch.concat(h_to_4h_weight[::-1], 0).numpy()

                ## Adaptation swiglu supports tp
                v = torch.from_numpy(v)
                w_split, v_split = torch.chunk(v, 2, dim=0)
                w_split = torch.chunk(w_split, mapping.tp_size, dim=0)
                v_split = torch.chunk(v_split, mapping.tp_size, dim=0)
                v_tmp = [torch.cat(weights, dim=0) for weights in zip(w_split, v_split)]

                dst = tensorrt_llm_model.encoder.layers[idx].mlp.fc.weight
                #split_v = split(v, mapping.tp_size, mapping.rank, dim = 0)
                split_v = v_tmp[mapping.rank]
                split_v = split_v.numpy()

                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_model.encoder.layers[
                        idx].mlp.fc.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.dense_4h_to_h.weight' in k:
                #v = v.transpose()
                dst = tensorrt_llm_model.encoder.layers[idx].mlp.proj.weight
                split_v = split(v, mapping.tp_size, mapping.rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_model.encoder.layers[
                        idx].mlp.proj.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')

    return tensorrt_llm_model

"""
def load_from_hf_chatglm2_6B(tensorrt_llm_model,
                             hf_model,
                             rank=0,
                             tensor_parallel=1,
                             dtype="float32",
                             multi_query_mode=False):
    tensorrt_llm.logger.info('Loading weights from HF ChatGLM2...')
    time.time()

    quant_mode = getattr(tensorrt_llm_model, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    str_dtype_to_torch(dtype)
    tensorrt_llm_model.encoder.final_layernorm.weight.value = hf_model.transformer.encoder.final_layernorm.weight.detach(
    ).cpu().numpy()
    tensorrt_llm_model.embedding.weight.value = hf_model.transformer.embedding.word_embeddings.weight.detach(
    ).cpu().numpy()
    tensorrt_llm_model.lm_head.weight.value = hf_model.transformer.output_layer.weight.detach(
    ).cpu().numpy()

    def load_quant_weight(src, value_dst, scale_dst,
                          plugin_weight_only_quant_type):
        v = np.ascontiguousarray(src.transpose())
        processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
            torch.tensor(v), plugin_weight_only_quant_type)
        # workaround for trt not supporting int8 inputs in plugins currently
        value_dst.value = processed_torch_weights.view(
            dtype=torch.float32).numpy()
        scale_dst.value = torch_weight_scales.numpy()

    for i in range(28):
        tensorrt_llm_model.encoder.layers[
            i].input_layernorm.weight.value = hf_model.transformer.encoder.layers[
                i].input_layernorm.weight.detach().cpu().numpy()
        tensorrt_llm_model.encoder.layers[
            i].post_attention_layernorm.weight.value = hf_model.transformer.encoder.layers[
                i].post_attention_layernorm.weight.detach().cpu().numpy()
        tensorrt_llm_model.encoder.layers[
            i].self_attention.qkv.bias.value = hf_model.transformer.encoder.layers[
                i].self_attention.query_key_value.bias.detach().cpu().numpy()
        # swap first and secont half weight columns to adapt trt_llm Swiglu
        h_to_4h_weight = hf_model.transformer.encoder.layers[
            i].mlp.dense_h_to_4h.weight.detach().cpu()
        h_to_4h_weight = torch.split(h_to_4h_weight,
                                     h_to_4h_weight.shape[0] // 2, 0)
        h_to_4h_weight = torch.concat(h_to_4h_weight[::-1], 0).numpy()
        if use_weight_only:

            load_quant_weight(
                src=h_to_4h_weight,
                value_dst=tensorrt_llm_model.encoder.layers[i].mlp.fc.weight,
                scale_dst=tensorrt_llm_model.encoder.layers[i].mlp.fc.
                per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            load_quant_weight(
                src=hf_model.transformer.encoder.layers[i].mlp.dense_4h_to_h.
                weight.detach().cpu().numpy(),
                value_dst=tensorrt_llm_model.encoder.layers[i].mlp.proj.weight,
                scale_dst=tensorrt_llm_model.encoder.layers[i].mlp.proj.
                per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            load_quant_weight(
                src=hf_model.transformer.encoder.layers[i].self_attention.
                query_key_value.weight.detach().cpu().numpy(),
                value_dst=tensorrt_llm_model.encoder.layers[i].self_attention.
                qkv.weight,
                scale_dst=tensorrt_llm_model.encoder.layers[i].self_attention.
                qkv.per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            load_quant_weight(
                src=hf_model.transformer.encoder.layers[i].self_attention.dense.
                weight.detach().cpu().numpy(),
                value_dst=tensorrt_llm_model.encoder.layers[i].self_attention.
                dense.weight,
                scale_dst=tensorrt_llm_model.encoder.layers[i].self_attention.
                dense.per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)

        else:
            tensorrt_llm_model.encoder.layers[
                i].self_attention.qkv.weight.value = hf_model.transformer.encoder.layers[
                    i].self_attention.query_key_value.weight.detach().cpu(
                    ).numpy()
            tensorrt_llm_model.encoder.layers[
                i].self_attention.dense.weight.value = hf_model.transformer.encoder.layers[
                    i].self_attention.dense.weight.detach().cpu().numpy()
            tensorrt_llm_model.encoder.layers[
                i].mlp.fc.weight.value = h_to_4h_weight
            tensorrt_llm_model.encoder.layers[
                i].mlp.proj.weight.value = hf_model.transformer.encoder.layers[
                    i].mlp.dense_4h_to_h.weight.detach().cpu().numpy()
    
    return tensorrt_llm_model
"""

if __name__ == '__main__':
    from tensorrt_llm.layers.attention import PositionEmbeddingType
    from tensorrt_llm.models import weight_only_quantize
    from tensorrt_llm.quantization import QuantMode

    kv_dtype = 'float16'
    quant_mode = QuantMode.use_weight_only(False)
    tensorrt_llm_ChatGLM2_6BModel = ChatGLM2HeadModel(
        num_layers=28,
        num_heads=32,
        hidden_size=4096,
        inter_size=None,
        vocab_size=65024,
        hidden_act='swiglu',
        max_position_embeddings=4096,
        position_embedding_type=PositionEmbeddingType.learned_absolute,
        rotary_embedding_percentage=1.0,
        dtype=kv_dtype,
        tensor_parallel=1,  # TP only
        tensor_parallel_group=list(range(1)),  # TP only
        apply_query_key_layer_scaling=False,
        quant_mode=quant_mode,
        bias=False,
        multi_query_mode=False)
    tensorrt_llm_ChatGLM2_6BModel = weight_only_quantize(
        tensorrt_llm_ChatGLM2_6BModel, quant_mode)

    model_dir = './pyTorchModel'

    print(f'Loading HF Chat_GLM2 ... from {model_dir}')

    import transformers
    hf_model = transformers.AutoModel.from_pretrained(
        model_dir, trust_remote_code=True).cpu()

    load_from_hf_chatglm2_6B(tensorrt_llm_ChatGLM2_6BModel,
                             hf_model,
                             0,
                             1,
                             dtype='float16',
                             multi_query_mode=False)
    del hf_model
