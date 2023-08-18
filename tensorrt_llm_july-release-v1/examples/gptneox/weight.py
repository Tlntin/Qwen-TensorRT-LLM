import time
from operator import attrgetter

import torch

import tensorrt_llm
from tensorrt_llm.models import GPTNeoXForCausalLM


def load_from_hf_gpt_neox(tensorrt_llm_gpt_neox: GPTNeoXForCausalLM,
                          hf_gpt_neox,
                          fp16=False):

    hf_model_gptneox_block_names = [
        "input_layernorm.weight",
        "input_layernorm.bias",
        "post_attention_layernorm.weight",
        "post_attention_layernorm.bias",
        "mlp.dense_h_to_4h.weight",
        "mlp.dense_h_to_4h.bias",
        "mlp.dense_4h_to_h.weight",
        "mlp.dense_4h_to_h.bias",
    ]

    tensorrt_llm_model_gptneox_block_names = [
        "input_layernorm.weight",
        "input_layernorm.bias",
        "post_attention_layernorm.weight",
        "post_attention_layernorm.bias",
        "mlp.fc.weight",
        "mlp.fc.bias",
        "mlp.proj.weight",
        "mlp.proj.bias",
    ]

    tensorrt_llm.logger.info('Loading weights from HF GPT-NeoX...')
    tik = time.time()

    torch_dtype = torch.float16 if fp16 else torch.float32
    hf_gpt_neox_state_dict = hf_gpt_neox.state_dict()

    v = hf_gpt_neox_state_dict.get('gpt_neox.embed_in.weight')
    tensorrt_llm_gpt_neox.embedding.weight.value = v.to(
        torch_dtype).cpu().numpy()

    n_layer = hf_gpt_neox.config.num_hidden_layers

    for layer_idx in range(n_layer):
        prefix = "gpt_neox.layers." + str(layer_idx) + "."
        for idx, hf_attr in enumerate(hf_model_gptneox_block_names):
            v = hf_gpt_neox_state_dict.get(prefix + hf_attr)
            layer = attrgetter(tensorrt_llm_model_gptneox_block_names[idx])(
                tensorrt_llm_gpt_neox.layers[layer_idx])
            setattr(layer, 'value', v.to(torch_dtype).cpu().numpy())

        # Attention QKV Linear
        # qkv_weights [num_heads x (q|k|v), hidden_size] ->
        # [(num_heads x q)|(num_heads x k)|(num_heads x v), hidden_size]
        qkv_weights = hf_gpt_neox_state_dict.get(
            prefix + "attention.query_key_value.weight")
        qkv_bias = hf_gpt_neox_state_dict.get(prefix +
                                              "attention.query_key_value.bias")

        num_heads = hf_gpt_neox.config.num_attention_heads
        hidden_size = hf_gpt_neox.config.hidden_size
        head_size = hidden_size // num_heads

        new_qkv_weight_shape = torch.Size(
            [num_heads, 3, head_size * qkv_weights.size()[-1]])
        new_qkv_bias_shape = torch.Size([num_heads, 3, head_size])

        qkv_weights = qkv_weights.view(new_qkv_weight_shape).permute(
            1, 0, 2).reshape([hidden_size * 3, hidden_size])
        qkv_bias = qkv_bias.view(new_qkv_bias_shape).permute(1, 0, 2).reshape(
            [hidden_size * 3])

        tensorrt_llm_gpt_neox.layers[layer_idx].attention.qkv.weight.value = \
          qkv_weights.to(torch_dtype).cpu().numpy()
        tensorrt_llm_gpt_neox.layers[layer_idx].attention.qkv.bias.value = \
          qkv_bias.to(torch_dtype).cpu().numpy()

        # Attention Dense Linear
        v = hf_gpt_neox_state_dict.get(prefix + "attention.dense.weight")
        tensorrt_llm_gpt_neox.layers[layer_idx].attention.dense.weight.value = \
          v.to(torch_dtype).cpu().numpy()
        v = hf_gpt_neox_state_dict.get(prefix + "attention.dense.bias")
        tensorrt_llm_gpt_neox.layers[layer_idx].attention.dense.bias.value = \
          v.to(torch_dtype).cpu().numpy()

    v = hf_gpt_neox_state_dict.get('gpt_neox.final_layer_norm.weight')
    tensorrt_llm_gpt_neox.ln_f.weight.value = v.to(torch_dtype).cpu().numpy()

    v = hf_gpt_neox_state_dict.get('gpt_neox.final_layer_norm.bias')
    tensorrt_llm_gpt_neox.ln_f.bias.value = v.to(torch_dtype).cpu().numpy()

    v = hf_gpt_neox_state_dict.get('embed_out.weight')
    tensorrt_llm_gpt_neox.lm_head.weight.value = v.to(torch_dtype).cpu().numpy()

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
