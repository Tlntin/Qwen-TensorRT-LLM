from ...layers import ColumnLinear, RowLinear
from ...models import GPTLMHeadModel
from ...quantization import (SmoothQuantAttention, SmoothQuantLayerNorm,
                             SmoothQuantMLP, WeightOnlyQuantColumnLinear,
                             WeightOnlyQuantRowLinear)


def smooth_quantize(model, quant_mode):
    assert isinstance(model,
                      GPTLMHeadModel), "Only GPTLMHeadModel is well tested now"
    assert quant_mode.has_act_and_weight_quant()
    for layer in model.layers:
        assert hasattr(layer,
                       "input_layernorm"), "The layer has no input_layernorm"
        layer.input_layernorm = SmoothQuantLayerNorm(
            normalized_shape=layer.hidden_size,
            dtype=layer.dtype,
            quant_mode=quant_mode)
        assert hasattr(layer, "attention"), "The layer has no attention"
        layer.attention = SmoothQuantAttention(
            layer.hidden_size,
            layer.num_attention_heads,
            layer.max_position_embeddings,
            layer.num_layers,
            layer.apply_query_key_layer_scaling,
            dtype=layer.dtype,
            attention_mask_type=layer.attention_mask_type,
            position_embedding_type=layer.position_embedding_type,
            tp_group=layer.tp_group,
            tp_size=layer.tp_size,
            quant_mode=quant_mode)
        assert hasattr(layer, "mlp"), "The layer has no mlp"
        layer.mlp = SmoothQuantMLP(hidden_size=layer.hidden_size,
                                   ffn_hidden_size=layer.hidden_size * 4,
                                   hidden_act=layer.hidden_act,
                                   dtype=layer.dtype,
                                   tp_group=layer.tp_group,
                                   tp_size=layer.tp_size,
                                   quant_mode=quant_mode)
        assert hasattr(layer,
                       "post_layernorm"), "The layer has no post_layernorm"
        layer.post_layernorm = SmoothQuantLayerNorm(
            normalized_shape=layer.hidden_size,
            dtype=layer.dtype,
            quant_mode=quant_mode)

    setattr(model, 'quant_mode', quant_mode)
    return model


def weight_only_quantize(model,
                         quant_mode,
                         exclude_modules=None,
                         current_key_name=None):
    assert quant_mode.is_weight_only()

    exclude_modules = ['lm_head'
                       ] if exclude_modules is None else exclude_modules

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if len(list(module.children())) > 0:
            weight_only_quantize(module, quant_mode, exclude_modules,
                                 current_key_name)

        if isinstance(module, ColumnLinear) and name not in exclude_modules:
            if not any(key in '.'.join(current_key_name)
                       for key in exclude_modules):
                model._modules[name] = WeightOnlyQuantColumnLinear(
                    in_features=module.in_features,
                    out_features=module.out_features * module.tp_size,
                    bias=module.bias is not None,
                    dtype=module.dtype,
                    tp_group=module.tp_group,
                    tp_size=module.tp_size,
                    gather_output=module.gather_output,
                    quant_mode=quant_mode)
        elif isinstance(module, RowLinear) and name not in exclude_modules:
            if not any(key in '.'.join(current_key_name)
                       for key in exclude_modules):
                model._modules[name] = WeightOnlyQuantRowLinear(
                    in_features=module.in_features * module.tp_size,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    dtype=module.dtype,
                    tp_group=module.tp_group,
                    tp_size=module.tp_size,
                    quant_mode=quant_mode)

        current_key_name.pop(-1)

    setattr(model, 'quant_mode', quant_mode)

    return model
