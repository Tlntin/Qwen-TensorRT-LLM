from typing import List, Optional, Sequence, Tuple, Union
import os
import numpy as np
import tensorrt as trt
import math
from tensorrt_llm.functional import ACT2FN
from tensorrt_llm.quantization.functional import quantize_per_token, quantize_tensor
from tensorrt_llm._common import default_net, default_trtnet, precision
from tensorrt_llm.plugin import  _TRT_LLM_PLUGIN_NAMESPACE as TRT_LLM_PLUGIN_NAMESPACE
from tensorrt_llm._utils import str_dtype_to_trt, str_dtype_to_np
from tensorrt_llm.parameter import Parameter
from tensorrt_llm.quantization.mode import QuantMode
from tensorrt_llm.quantization.layer import (
    SmoothQuantColumnLinear, SmoothQuantRowLinear
)
from tensorrt_llm.layers.attention import AttentionMaskType, PositionEmbeddingType
from tensorrt_llm.module import Module
from tensorrt_llm.functional import (
    ACT2FN, RaggedTensor, Tensor, cast, concat, constant, gpt_attention,
    shape, slice, _create_tensor, expand_dims_like, split
)



# copy from quantization/layer.py SmoothQuantAttention
# but i add some layer to it, just like QWenAttention in model.py
class SmoothQuantAttention(Module):

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        max_position_embeddings, # 8192
        seq_length, # 2048
        num_kv_heads=None,
        num_layers=1,
        apply_query_key_layer_scaling=False,
        attention_mask_type=AttentionMaskType.causal,
        bias=True,
        dtype=None,
        position_embedding_type=PositionEmbeddingType.rope,
        tp_group=None,
        tp_size=1,
        multi_block_mode=False,
        multi_query_mode=False,
        use_dynamic_ntk=True,
        use_logn_attn=True,
        paged_kv_cache=False,
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
        #if self.position_embedding_type.is_rope():
        self.rotary_embedding_dim = hidden_size // num_attention_heads

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
            bias=bias,
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
        self.use_dynamic_ntk = use_dynamic_ntk
        self.use_logn_attn = use_logn_attn

        logn_array = np.array([
            math.log(i, seq_length) if i > seq_length else 1
            for i in range(1, 32768)
            ],
            dtype=np.float16
        ).reshape(1, -1, 1, 1)
        self.logn_tensor = Parameter(
            value=logn_array,
            dtype=self.dtype,
            shape=[1, 32767, 1, 1],
        ) 

    def forward(
        self,
        hidden_states: Union[Tensor, RaggedTensor],
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
        if isinstance(hidden_states, RaggedTensor):
            input_lengths = hidden_states.row_lengths
            max_input_length = hidden_states.max_row_length
            hidden_states = hidden_states.data
            is_input_ragged_tensor = True
        if default_net().plugin_config.gpt_attention_plugin:
            qkv = self.qkv(hidden_states)
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

            kv_orig_quant_scale = self.kv_orig_quant_scale.value \
                if self.quant_mode.has_int8_kv_cache() else None
            kv_quant_orig_scale = self.kv_quant_orig_scale.value \
                if self.quant_mode.has_int8_kv_cache() else None

            # implement in trt
            seq_start = slice(shape(key), [1], [1]) - slice(shape(query), [1], [1])
            seq_end = slice(shape(key), [1], [1])
            logn_shape = self.logn_tensor.value.shape
            logn_tensor = slice(
                input=self.logn_tensor.value,
                starts=concat([0, seq_start, 0, 0]),
                sizes=concat([logn_shape[0], seq_end - seq_start, logn_shape[2], logn_shape[3]]),
            )
            query = query * expand_dims_like(logn_tensor, query)

            qkv = concat([query, key, value], dim=2)
            qkv = qkv.view(
                concat([shape(qkv, 0),
                        shape(qkv, 1),
                        self.hidden_size * 3])
            )
        else:
            raise ValueError("gpt_attention_plugin is not set")
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
                past_key_value_length=past_key_value_length,
                masked_tokens=masked_tokens,
                input_lengths=input_lengths,
                max_input_length=max_input_length,
                cache_indirection=cache_indirection,
                num_heads=self.num_attention_heads,
                head_size=self.num_kv_heads,
                q_scaling=self.q_scaling,
                rotary_embedding_dim=0, # self.rotary_embedding_dim,
                neox_rotary_style=self.position_embedding_type,
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
            raise Exception("gpt_attention_plugin is not set")

        if self.quant_mode.has_act_and_weight_quant():
            if self.quant_mode.has_act_static_scaling():
                context = quantize_tensor(
                    context, self.quantization_scaling_factor.value)
            else:
                context = quantize_per_token(context)

        context = self.dense(context)
        if is_input_ragged_tensor:
            context = RaggedTensor.from_row_lengths(context, input_lengths,
                                                    max_input_length)

        if use_cache:
            return (context, past_key_value)

        return context


# copy from quantization/layer.py SmoothQuantMLP
# but i add some layer to it
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
# Todo: build a new plugin for rmsnorm
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
        layer.ln_2 = SmoothQuantRmsNorm(
            normalized_shape=layer.hidden_size,
            plugin_dtype=rmsnorm_quantization_plugin_dtype,
            custom_plugin_paths=custom_plugin_paths,
            dtype=layer.dtype,
            quant_mode=quant_mode
        )

    setattr(model, 'quant_mode', quant_mode)
    return model

