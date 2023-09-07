from tensorrt_llm.functional import ACT2FN
from tensorrt_llm.quantization.functional import quantize_per_token, quantize_tensor
from tensorrt_llm.quantization import SmoothQuantAttention, SmoothQuantLayerNorm
from tensorrt_llm.parameter import Parameter
from tensorrt_llm.quantization.mode import QuantMode
from tensorrt_llm.quantization.layer import (
    SmoothQuantColumnLinear, SmoothQuantRowLinear
)
from tensorrt_llm.module import Module



class SmoothQuantMLP(Module):

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 hidden_act,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 quant_mode=QuantMode(0)):
        super().__init__()
        if hidden_act not in ACT2FN:
            raise ValueError(
                'unsupported activation function: {}'.format(hidden_act))
        self.w1 = SmoothQuantColumnLinear(
            hidden_size,
            ffn_hidden_size,
            bias=bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            gather_output=False,
            quant_mode=quant_mode
        )

        self.w2 = SmoothQuantColumnLinear(
            hidden_size,
            ffn_hidden_size,
            bias=bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            gather_output=False,
            quant_mode=quant_mode
        )

        self.c_proj = SmoothQuantRowLinear(
            ffn_hidden_size,
            hidden_size,
            bias=bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=quant_mode
        )

        self.hidden_act = hidden_act
        self.quant_mode = quant_mode

        if self.quant_mode.has_act_static_scaling():
            self.quantization_scaling_factor = Parameter(shape=(1, ),
                                                         dtype='float32')
        else:
            self.register_parameter('quantization_scaling_factor', None)

    def forward(self, hidden_states):
        a1 = self.w1(hidden_states)
        a2 = self.w2(hidden_states)
        inter = a1 * ACT2FN[self.hidden_act](a2)
        if self.quant_mode.has_act_and_weight_quant():
            if self.quant_mode.has_act_static_scaling():
                # Avoid quantiztion layers as it breaks int8 plugins
                inter = quantize_tensor(
                    inter,
                    self.quantization_scaling_factor.value
                )
            else:
                # Quantize per token outputs tuple:
                # quantized tensor and scaling factors per token
                inter = quantize_per_token(inter)
        output = self.proj(inter)
        return output


def smooth_quantize(model, quant_mode):
    assert quant_mode.has_act_and_weight_quant()
    for layer in model.layers:
        assert hasattr(layer, "ln_1"), "The layer has no ln_1"
        layer.ln_1 = SmoothQuantLayerNorm(
            normalized_shape=layer.hidden_size,
            dtype=layer.dtype,
            quant_mode=quant_mode
        )
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
            quant_mode=quant_mode
        )
        assert hasattr(layer, "mlp"), "The layer has no mlp"
        layer.mlp = SmoothQuantMLP(
            hidden_size=layer.hidden_size,
            ffn_hidden_size=layer.mlp_hidden_size // 2,
            hidden_act=layer.hidden_act,
            dtype=layer.dtype,
            tp_group=layer.tp_group,
            tp_size=layer.tp_size,
            quant_mode=quant_mode
        )
        assert hasattr(layer, "ln_2"), "The layer has no ln_2"
        layer.ln_2 = SmoothQuantLayerNorm(
            normalized_shape=layer.hidden_size,
            dtype=layer.dtype,
            quant_mode=quant_mode
        )

    setattr(model, 'quant_mode', quant_mode)
    return model

