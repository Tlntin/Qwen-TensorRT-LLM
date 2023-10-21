from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorrt as trt
import math
from tensorrt_llm.functional import ACT2FN
from tensorrt_llm.quantization.functional import quantize_per_token, quantize_tensor
from tensorrt_llm._common import default_net, default_trtnet, precision
from tensorrt_llm.plugin import  TRT_LLM_PLUGIN_NAMESPACE
from tensorrt_llm._utils import str_dtype_to_trt, str_dtype_to_np
from tensorrt_llm.parameter import Parameter
from tensorrt_llm.quantization.mode import QuantMode
from tensorrt_llm.quantization.layers import (
    SmoothQuantColumnLinear, smooth_quant_gemm, SmoothQuantRowLinear
)
from tensorrt_llm.layers.attention import AttentionMaskType, PositionEmbeddingType
from tensorrt_llm.module import Module
from tensorrt_llm.functional import (ACT2FN, Tensor, allgather, allreduce,
                          cast, concat, constant, gpt_attention, matmul, mul,
                          shape, slice, softmax, split, expand_dims_like, where, _create_tensor)


class SmoothQuantAttention(Module):

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        max_position_embeddings,
        seq_length,
        num_layers=1,
        num_kv_heads=None,
        apply_query_key_layer_scaling=False,
        attention_mask_type=AttentionMaskType.causal,
        bias=False,
        dtype=None,
        position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
        neox_rotary_style=False,
        tp_group=None,
        tp_size=1,
        multi_block_mode=False,
        multi_query_mode=False,
        paged_kv_cache=False,
        use_dynamic_ntk=True,
        use_logn_attn=True,
        rotary_embedding_percentage=1.0,
        quant_mode=QuantMode(0)
    ):
        super().__init__()
        self.attention_mask_type = attention_mask_type
        self.attention_head_size = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads // tp_size
        self.num_kv_heads = (
            num_kv_heads + tp_size - 1
        ) // tp_size if num_kv_heads is not None else self.num_attention_heads
        self.hidden_size = hidden_size // tp_size
        self.split_size = hidden_size
        self.max_position_embeddings = max_position_embeddings

        self.num_layers = num_layers
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.q_scaling = 1
        if self.apply_query_key_layer_scaling:
            self.norm_factor *= self.num_layers
            self.q_scaling *= self.num_layers

        self.position_embedding_type = position_embedding_type
        self.multi_block_mode = multi_block_mode
        self.multi_query_mode = multi_query_mode
        self.paged_kv_cache = paged_kv_cache

        self.rotary_embedding_dim = 0
        self.neox_rotary_style = neox_rotary_style
        #if self.position_embedding_type.is_rope():
        if self.position_embedding_type == PositionEmbeddingType.rope:
            self.rotary_embedding_dim = int(self.attention_head_size *
                                            rotary_embedding_percentage)
            # TODO: Once we add RotaryEmbedding outside GPTAttention plugin,
            #       we need to set it up here

        self.quant_mode = quant_mode
        self.dtype = dtype

        if self.quant_mode.has_act_static_scaling():
            self.quantization_scaling_factor = Parameter(shape=(1, ),
                                                         dtype='float32')
        else:
            self.register_parameter('quantization_scaling_factor', None)

        qkv_quant_mode = quant_mode
        if self.quant_mode.has_act_and_weight_quant():
            # We need to hijack quant_mode for QKV because QKV always uses per channel scaling
            qkv_quant_mode = QuantMode.from_description(
                True, True, quant_mode.has_per_token_dynamic_scaling(), True)

        if self.quant_mode.has_int8_kv_cache():
            self.kv_orig_quant_scale = Parameter(shape=(1, ), dtype='float32')
            self.kv_quant_orig_scale = Parameter(shape=(1, ), dtype='float32')
        else:
            self.register_parameter('kv_orig_quant_scale', None)
            self.register_parameter('kv_quant_orig_scale', None)

        self.qkv = SmoothQuantColumnLinear(
            hidden_size,
            hidden_size * 3 if not multi_query_mode 
            else hidden_size + 2 * self.num_kv_heads * tp_size * self.attention_head_size,
            bias=True,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            gather_output=False,
            quant_mode=qkv_quant_mode)

        self.dense = SmoothQuantRowLinear(
            hidden_size,
            hidden_size,
            bias=bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=quant_mode
        )
        
        # copy from model.py QWenAttention
        # for input_length < 2048, logn and ntk is useless
        # self.use_dynamic_ntk = use_dynamic_ntk
        # self.use_logn_attn = use_logn_attn

        # logn_array = np.array([
        #     math.log(i, seq_length) if i > seq_length else 1
        #     for i in range(1, 32768)
        #     ],
        #     dtype=np.float16
        # ).reshape(1, -1, 1, 1)
        # self.logn_tensor = Parameter(
        #     value=logn_array,
        #     dtype=self.dtype,
        #     shape=[1, 32767, 1, 1],
        # ) 

    def forward(
        self,
        hidden_states: Union[Tensor, Tensor],
        rotary_pos_emb,
        past_key_value,
        sequence_length,
        past_key_value_length,
        masked_tokens,
        cache_indirection,
        use_cache=False,
        kv_cache_block_pointers=None,
        host_input_lengths=None,
        host_request_types=None,
    ):
        # TODO(nkorobov) add in-flight batching to SmoothQuant
        is_input_ragged_tensor = False
        assert isinstance(hidden_states, Tensor)
        if default_net().plugin_config.smooth_quant_gemm_plugin:
            qkv = self.qkv(hidden_states)
        else:
            raise ValueError("smooth_quant_gemm_plugin is not set")
        """
        # copy from model.py QWenAttention
        # query, key, value = qkv.split(self.split_size, dim=2)
        query, key, value = split(qkv, self.split_size, dim=2)
        # query = self._split_heads(query, self.num_heads, self.head_dim)
        query = query.view(
            concat([
                shape(qkv, 0),
                shape(qkv, 1),
                self.num_attention_heads,
                self.attention_head_size
            ]))
        # key = self._split_heads(key, self.num_heads, self.head_dim)
        key = key.view(
            concat([
                shape(qkv, 0),
                shape(qkv, 1),
                self.num_attention_heads,
                self.attention_head_size
            ]))
        # value = self._split_heads(value, self.num_heads, self.head_dim)
        value = value.view(
            concat([
                shape(qkv, 0),
                shape(qkv, 1),
                self.num_attention_heads,
                self.attention_head_size
            ]))

        zero = cast(constant(
            np.ascontiguousarray(
                np.zeros(
                    [1, 1, 1, 1],
                    dtype=np.float16 if self.dtype == trt.float16 else np.float32
                )
            )
        ), dtype=trt.float32)
        def _rotate_half(x128):
            x64_part0, x64_part1 = x128.split(64, dim=-1)

            x64_part1_negtive = zero - x64_part1

            y64 = concat([x64_part1_negtive, x64_part0], dim=3)
            return y64

        def apply_rotary_pos_emb(t, freqs):
            cos1, sin1 = freqs
            t_ = t.cast(trt.float32)
            t_rotate = _rotate_half(t_)
            y128 = t_ * cos1 + t_rotate * sin1
            # y128 = y128.view(shape(x))
            y128 = y128.cast(t.dtype)
            return y128
        q_pos_emb, k_pos_emb = rotary_pos_emb
        query = apply_rotary_pos_emb(query, q_pos_emb)
        key = apply_rotary_pos_emb(key, k_pos_emb)
        # implement in trt
        # seq_start = slice(shape(key), [1], [1]) - slice(shape(query), [1], [1])
        # seq_end = slice(shape(key), [1], [1])
        # logn_shape = self.logn_tensor.value.shape
        # logn_tensor = slice(
        #     input=self.logn_tensor.value,
        #     starts=concat([0, seq_start, 0, 0]),
        #     sizes=concat([logn_shape[0], seq_end - seq_start, logn_shape[2], logn_shape[3]]),
        # )
        # query = query * expand_dims_like(logn_tensor, query)

        qkv = concat([query, key, value], dim=2)
        qkv = qkv.view(
            concat([shape(qkv, 0),
                    shape(qkv, 1),
                    self.hidden_size * 3])
        )
        """ 
        kv_orig_quant_scale = self.kv_orig_quant_scale.value \
            if self.quant_mode.has_int8_kv_cache() else None
        kv_quant_orig_scale = self.kv_quant_orig_scale.value \
            if self.quant_mode.has_int8_kv_cache() else None

        
        if default_net().plugin_config.gpt_attention_plugin:
            assert sequence_length is not None
            assert past_key_value_length is not None
            assert self.attention_mask_type == AttentionMaskType.causal, \
                'Plugin only support masked MHA.'
            # assert host_request_types is not None
            if default_net().plugin_config.remove_input_padding:
                assert host_input_lengths is not None

            context, past_key_value = gpt_attention(
                tensor=qkv,
                past_key_value=past_key_value,
                sequence_length=sequence_length,
                past_key_value_lengths=past_key_value_length,
                masked_tokens=masked_tokens,
                cache_indirection=cache_indirection,
                num_heads=self.num_attention_heads,
                head_size=self.attention_head_size,
                q_scaling=self.q_scaling,
                rotary_embedding_dim=self.rotary_embedding_dim, # when we use it 0, we will not use rotary embedding in plugin
                neox_rotary_style=self.neox_rotary_style,
                multi_block_mode=self.multi_block_mode,
                multi_query_mode=self.multi_query_mode,
                kv_orig_quant_scale=kv_orig_quant_scale,
                kv_quant_orig_scale=kv_quant_orig_scale,
                use_int8_kv_cache=self.quant_mode.has_int8_kv_cache(),
                kv_cache_block_pointers=kv_cache_block_pointers,
                host_input_lengths=host_input_lengths,
                host_request_types=host_request_types,
            )
        else:
            assert self.paged_kv_cache == False

            def transpose_for_scores(x):
                new_x_shape = concat([
                    shape(x, 0),
                    shape(x, 1), self.num_attention_heads,
                    self.attention_head_size
                ])
                return x.view(new_x_shape).permute([0, 2, 1, 3])

            query, key, value = split(qkv, self.hidden_size, dim=2)
            query = transpose_for_scores(query)
            key = transpose_for_scores(key)
            value = transpose_for_scores(value)

            if past_key_value is not None:

                def dequantize_tensor(x, scale):
                    # Cast from int8 to dtype
                    casted_x = cast(x, self.dtype)
                    return casted_x * scale

                if self.quant_mode.has_int8_kv_cache():
                    past_key_value = dequantize_tensor(
                        past_key_value, self.kv_dequantization_scale.value)

                # past_key_value [bs, 2, num_heads, max_seq_len, head_dim]
                past_key, past_value = split(past_key_value, 1, dim=1)

                key_shape = concat([
                    shape(past_key, 0),
                    shape(past_key, 2),
                    shape(past_key, 3),
                    shape(past_key, 4)
                ])
                past_key = past_key.view(key_shape, zero_is_placeholder=False)
                past_value = past_value.view(key_shape,
                                             zero_is_placeholder=False)
                key = concat([past_key, key], dim=2)
                value = concat([past_value, value], dim=2)

            def merge_caches():
                key_inflated_shape = concat([
                    shape(key, 0), 1,
                    shape(key, 1),
                    shape(key, 2),
                    shape(key, 3)
                ])
                inflated_key = key.view(key_inflated_shape,
                                        zero_is_placeholder=False)
                inflated_value = value.view(key_inflated_shape,
                                            zero_is_placeholder=False)
                past_key_value = concat([inflated_key, inflated_value], dim=1)
                return past_key_value

            if self.attention_mask_type == AttentionMaskType.causal:
                query_length = shape(query, 2)
                key_length = shape(key, 2)
                starts = concat([0, 0, key_length - query_length, 0])
                sizes = concat([1, 1, query_length, key_length])
                buffer = constant(
                    np.expand_dims(
                        np.tril(
                            np.ones(
                                (self.max_position_embeddings,
                                 self.max_position_embeddings))).astype(bool),
                        (0, 1)))
                causal_mask = slice(buffer, starts, sizes)

            key = key.permute([0, 1, 3, 2])
            with precision('float32'):
                attention_scores = matmul(cast(query, 'float32'),
                                          cast(key, 'float32'))

                if self.attention_mask_type == AttentionMaskType.causal:
                    attention_scores = where(causal_mask, attention_scores,
                                             -10000.0)

                attention_scores = attention_scores / self.norm_factor
                attention_probs = softmax(attention_scores, dim=-1)

            context = matmul(attention_probs, value).permute([0, 2, 1, 3])
            context = context.view(
                concat([shape(context, 0),
                        shape(context, 1), self.hidden_size]))

            past_key_value = merge_caches()

            if use_cache and self.quant_mode.has_int8_kv_cache():
                past_key_value = quantize_tensor(
                    past_key_value, self.kv_quantization_scale.value)

        if self.quant_mode.has_act_and_weight_quant():
            if self.quant_mode.has_act_static_scaling():
                context = quantize_tensor(
                    context, self.quantization_scaling_factor.value)
            else:
                context = quantize_per_token(context)

        context = self.dense(context)

        if use_cache:
            return (context, past_key_value)

        return context

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
        output = self.c_proj(inter)
        return output


# copy from quantization/functional.py smooth_quant_layer_norm
def smooth_quant_rms_norm_op(
    input: Tensor,
    plugin_dtype: str,
    normalized_shape: Union[int, Tuple[int]],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    scale: Optional[Tensor] = None,
    eps: float = 1e-05,
    custom_plugin_paths = None,
    dynamic_act_scaling: bool = False
) -> Tensor:
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    if custom_plugin_paths is None:
        custom_plugin_paths = []
    # create plugin
    if len(custom_plugin_paths) > 0:
        plugin_creator = trt.get_plugin_registry().get_plugin_creator(
            'RmsnormQuantization', '1'
        )
    else:
        plugin_creator = trt.get_plugin_registry().get_plugin_creator(
            'RmsnormQuantization', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plugin_creator is not None

    eps = trt.PluginField(
        "eps",
        np.array(eps, dtype=np.float32),
        trt.PluginFieldType.FLOAT32
    )

    dyn_act_scaling = trt.PluginField(
        "dyn_act_scaling",
        np.array([int(dynamic_act_scaling)], np.int32),
        trt.PluginFieldType.INT32
    )

    plugin_fileds = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(plugin_dtype))], np.int32),
        trt.PluginFieldType.INT32)
    plugin_filed_collections = trt.PluginFieldCollection(
        [eps, dyn_act_scaling, plugin_fileds]
    )
    rmsnorm_plugin = plugin_creator.create_plugin(
        "rmsnorm_quantization", plugin_filed_collections
    )
    
    if weight is None:
        weight = constant(
            np.ones(normalized_shape, dtype=str_dtype_to_np(plugin_dtype)))
    if bias is None:
        bias = constant(
            np.zeros(normalized_shape, dtype=str_dtype_to_np(plugin_dtype)))

    inputs = [
        input.trt_tensor, weight.trt_tensor, bias.trt_tensor,
        scale.trt_tensor
    ]
    layer = default_trtnet().add_plugin_v2(inputs, rmsnorm_plugin)
    layer.get_output(0).set_dynamic_range(-127, 127)
    if not dynamic_act_scaling:
        return _create_tensor(layer.get_output(0), layer)

    return (
        _create_tensor(
            layer.get_output(0),
            layer
        ), 
        _create_tensor(
            layer.get_output(1),
            layer
        )
    )


# copy from quantization/layer.py SmoothQuantLayerNorm
class SmoothQuantRmsNorm(Module):

    def __init__(self,
                 normalized_shape,
                 plugin_dtype,
                 custom_plugin_paths=None,
                 eps=1e-06,
                 elementwise_affine=True,
                 dtype=None,
                 quant_mode=QuantMode(0),
                 bias=False):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        if not quant_mode.has_act_and_weight_quant():
            raise ValueError(
                "SmoothQuant Rms norm has to have some quantization mode set")
        self.plugin_dtype = plugin_dtype
        if custom_plugin_paths is None:
            custom_plugin_paths = []
        self.custom_plugin_paths = custom_plugin_paths
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(shape=self.normalized_shape, dtype=dtype)
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = Parameter(shape=self.normalized_shape, dtype=dtype)
        else:
            self.register_parameter('bias', None)

        self.eps = eps
        self.quant_mode = quant_mode

        if self.quant_mode.has_act_and_weight_quant():
            self.scale_to_int = Parameter(shape=(1, ), dtype=dtype)
        else:
            self.register_parameter('scale_to_int', None)

    def forward(self, x):
        weight = None if self.weight is None else self.weight.value
        bias = None if self.bias is None else self.bias.value
        scale = None if self.scale_to_int is None else self.scale_to_int.value
        return smooth_quant_rms_norm_op(
            x,
            self.plugin_dtype,
            self.normalized_shape,
            weight,
            bias,
            scale,
            self.eps,
            custom_plugin_paths=self.custom_plugin_paths,
            dynamic_act_scaling=self.quant_mode.has_per_token_dynamic_scaling()
        )


def smooth_quantize(model, quant_mode, rmsnorm_quantization_plugin_dtype, custom_plugin_paths=None):
    assert quant_mode.has_act_and_weight_quant()
    if custom_plugin_paths is None:
        custom_plugin_paths = []
    for layer in model.layers:
        assert hasattr(layer, "ln_1"), "The layer has no ln_1"
        layer.ln_1 = SmoothQuantRmsNorm(
            normalized_shape=layer.hidden_size,
            plugin_dtype=rmsnorm_quantization_plugin_dtype,
            custom_plugin_paths=custom_plugin_paths,
            dtype=layer.dtype,
            quant_mode=quant_mode
        )
        assert hasattr(layer, "attention"), "The layer has no attention"
        layer.attention = SmoothQuantAttention(
            layer.hidden_size,
            layer.num_attention_heads,
            max_position_embeddings=layer.max_position_embeddings,
            seq_length=layer.seq_length,
            num_layers=layer.num_layers,
            apply_query_key_layer_scaling=layer.apply_query_key_layer_scaling,
            attention_mask_type=layer.attention_mask_type,
            bias=layer.bias,
            dtype=layer.dtype,
            position_embedding_type=layer.position_embedding_type,
            neox_rotary_style=layer.neox_rotary_style,
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
            bias=layer.bias,
            tp_group=layer.tp_group,
            tp_size=layer.tp_size,
            quant_mode=quant_mode
        )
        assert hasattr(layer, "ln_2"), "The layer has no ln_2"
        layer.ln_2 = SmoothQuantRmsNorm(
            normalized_shape=layer.hidden_size,
            plugin_dtype=rmsnorm_quantization_plugin_dtype,
            custom_plugin_paths=custom_plugin_paths,
            dtype=layer.dtype,
            quant_mode=quant_mode
        )

    setattr(model, 'quant_mode', quant_mode)
    return model

