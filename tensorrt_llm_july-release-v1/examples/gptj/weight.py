import time
from operator import attrgetter

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm.models import GPTJForCausalLM
from tensorrt_llm.quantization import QuantMode


def load_from_hf_gpt_j(tensorrt_llm_gpt_j: GPTJForCausalLM,
                       hf_gpt_j,
                       fp16=False,
                       scaling_factors=None):

    hf_model_gptj_block_names = [
        "ln_1.weight",
        "ln_1.bias",
        "mlp.fc_in.weight",
        "mlp.fc_in.bias",
        "mlp.fc_out.weight",
        "mlp.fc_out.bias",
    ]

    tensorrt_llm_model_gptj_block_names = [
        "input_layernorm.weight",
        "input_layernorm.bias",
        "mlp.fc.weight",
        "mlp.fc.bias",
        "mlp.proj.weight",
        "mlp.proj.bias",
    ]

    quant_mode = getattr(tensorrt_llm_gpt_j, 'quant_mode', QuantMode(0))

    tensorrt_llm.logger.info('Loading weights from HF GPT-J...')
    tik = time.time()

    torch_dtype = torch.float16 if fp16 else torch.float32
    hf_gpt_j_state_dict = hf_gpt_j.state_dict()

    v = hf_gpt_j_state_dict.get('transformer.wte.weight')
    tensorrt_llm_gpt_j.embedding.weight.value = v.to(torch_dtype).cpu().numpy()

    n_layer = hf_gpt_j.config.n_layer

    for layer_idx in range(n_layer):
        prefix = "transformer.h." + str(layer_idx) + "."
        for idx, hf_attr in enumerate(hf_model_gptj_block_names):
            v = hf_gpt_j_state_dict.get(prefix + hf_attr)
            layer = attrgetter(tensorrt_llm_model_gptj_block_names[idx])(
                tensorrt_llm_gpt_j.layers[layer_idx])
            if idx == 2 and scaling_factors:
                tensorrt_llm_gpt_j.layers[
                    layer_idx].mlp.fc.activation_scaling_factor.value = np.array(
                        [scaling_factors['fc_act'][layer_idx]],
                        dtype=np.float32)

                tensorrt_llm_gpt_j.layers[
                    layer_idx].mlp.fc.weights_scaling_factor.value = np.array(
                        [scaling_factors['fc_weights'][layer_idx]],
                        dtype=np.float32)

            elif idx == 4 and scaling_factors:
                tensorrt_llm_gpt_j.layers[
                    layer_idx].mlp.proj.activation_scaling_factor.value = np.array(
                        [scaling_factors['proj_act'][layer_idx]],
                        dtype=np.float32)

                tensorrt_llm_gpt_j.layers[
                    layer_idx].mlp.proj.weights_scaling_factor.value = np.array(
                        [scaling_factors['proj_weights'][layer_idx]],
                        dtype=np.float32)
            setattr(layer, 'value', v.to(torch_dtype).cpu().numpy())

        # Attention QKV Linear
        # concatenate the Q, K, V layers weights.
        q_weights = hf_gpt_j_state_dict.get(prefix + "attn.q_proj.weight")
        k_weights = hf_gpt_j_state_dict.get(prefix + "attn.k_proj.weight")
        v_weights = hf_gpt_j_state_dict.get(prefix + "attn.v_proj.weight")
        qkv_weights = torch.cat((q_weights, k_weights, v_weights))
        layer = attrgetter("attention.qkv.weight")(
            tensorrt_llm_gpt_j.layers[layer_idx])
        setattr(layer, "value", qkv_weights.to(torch_dtype).cpu().numpy())
        if scaling_factors:
            tensorrt_llm_gpt_j.layers[
                layer_idx].attention.qkv.activation_scaling_factor.value = np.array(
                    [scaling_factors['qkv_act'][layer_idx]], dtype=np.float32)
            tensorrt_llm_gpt_j.layers[
                layer_idx].attention.qkv.weights_scaling_factor.value = np.array(
                    [scaling_factors['qkv_weights'][layer_idx]],
                    dtype=np.float32)

        if quant_mode.has_fp8_kv_cache():
            tensorrt_llm_gpt_j.layers[
                layer_idx].attention.kv_quantization_scale.value = np.array(
                    [scaling_factors['qkv_output'][layer_idx]],
                    dtype=np.float32)
            tensorrt_llm_gpt_j.layers[
                layer_idx].attention.kv_dequantization_scale.value = np.array(
                    [1.0 / scaling_factors['qkv_output'][layer_idx]],
                    dtype=np.float32)

        # Attention Dense (out_proj) Linear
        v = hf_gpt_j_state_dict.get(prefix + "attn.out_proj.weight")
        layer = attrgetter("attention.dense.weight")(
            tensorrt_llm_gpt_j.layers[layer_idx])
        setattr(layer, "value", v.to(torch_dtype).cpu().numpy())
        if scaling_factors:
            tensorrt_llm_gpt_j.layers[
                layer_idx].attention.dense.activation_scaling_factor.value = np.array(
                    [scaling_factors['dense_act'][layer_idx]], dtype=np.float32)
            tensorrt_llm_gpt_j.layers[
                layer_idx].attention.dense.weights_scaling_factor.value = np.array(
                    [scaling_factors['dense_weights'][layer_idx]],
                    dtype=np.float32)

    v = hf_gpt_j_state_dict.get('transformer.ln_f.weight')
    tensorrt_llm_gpt_j.ln_f.weight.value = v.to(torch_dtype).cpu().numpy()

    v = hf_gpt_j_state_dict.get('transformer.ln_f.bias')
    tensorrt_llm_gpt_j.ln_f.bias.value = v.to(torch_dtype).cpu().numpy()

    v = hf_gpt_j_state_dict.get('lm_head.weight')
    tensorrt_llm_gpt_j.lm_head.weight.value = v.to(torch_dtype).cpu().numpy()

    v = hf_gpt_j_state_dict.get('lm_head.bias')
    tensorrt_llm_gpt_j.lm_head.bias.value = v.to(torch_dtype).cpu().numpy()

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
