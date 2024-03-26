from collections import OrderedDict
import math
# import torch
from typing import Union, Optional, Tuple
import tensorrt as trt
import numpy as np
from tensorrt_llm._common import default_net, default_trtnet, precision
from tensorrt_llm._utils import pad_vocab_size, str_dtype_to_trt, \
    str_dtype_to_np
from tensorrt_llm.functional import (
    ACT2FN, Tensor, assertion, expand_mask, gather_last_token_logits,
    shape, concat, constant, gpt_attention, slice, expand_dims_like, cast, pow,
    _create_tensor, arange, outer, sin, cos, unary, partial, expand, recv, send,
    RotaryScalingType
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.generation_mixin import GenerationMixin
from tensorrt_llm.parameter import Parameter
from tensorrt_llm.layers import (
    AttentionParams, AttentionMaskType, ColumnLinear, Embedding,
    KeyValueCacheParams, PositionEmbeddingType, RowLinear, PromptTuningEmbedding, Attention
)
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.plugin import TRT_LLM_PLUGIN_NAMESPACE
from tensorrt_llm.quantization.layers import FP8Linear, FP8RowLinear

log = partial(unary, op=trt.UnaryOperation.LOG)
ceil = partial(unary, op=trt.UnaryOperation.CEIL)


class GPTEmbedding(Module):

    def __init__(self,
                 vocab_size,
                 hidden_size,
                 max_position_embeddings,
                 position_embedding_type=PositionEmbeddingType.learned_absolute,
                 dtype=None,
                 use_prompt_tuning=False,
                 tensor_parallel=1,
                 tensor_parallel_group=None,
                 sharding_dim=0,
                 tp_rank=None):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type
        self.use_prompt_tuning = use_prompt_tuning

        EmbeddingCls = PromptTuningEmbedding if use_prompt_tuning else Embedding
        self.vocab_embedding = EmbeddingCls(vocab_size,
                                            hidden_size,
                                            dtype=dtype,
                                            tp_size=tensor_parallel,
                                            tp_group=tensor_parallel_group,
                                            sharding_dim=sharding_dim,
                                            tp_rank=tp_rank)

        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            self.position_embedding = Embedding(max_position_embeddings,
                                                hidden_size,
                                                dtype=dtype)

    def forward(self,
                input_ids,
                position_ids,
                prompt_embedding_table=None,
                prompt_tasks=None,
                prompt_vocab_size=None):
        args = []
        if self.use_prompt_tuning:
            args = [prompt_embedding_table, prompt_tasks, prompt_vocab_size]
        x = self.vocab_embedding(input_ids, *args)
        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            x = x + self.position_embedding(position_ids)

        return x


def identity_op(tensor: Tensor) -> Tensor:
    input_tensor = tensor.trt_tensor
    # Create a plugin instance.
    plugin_creator = trt.get_plugin_registry().get_plugin_creator(
        'Identity', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plugin_creator is not None
    pfc = trt.PluginFieldCollection([])
    plugin = plugin_creator.create_plugin("identity", pfc)
    layer = default_trtnet().add_plugin_v2([input_tensor], plugin)
    return _create_tensor(layer.get_output(0), layer)


def rms_norm(input: Tensor,
             normalized_shape: Union[int, Tuple[int]],
             weight: Optional[Tensor] = None,
             eps: float = 1e-06) -> Tensor:
    '''
    Add a RMS norm operation on a tensor.
    Copy from tensorrt_llm.functional, for reduce some warning;

    TODO: Document!
    '''
    normalized_shape = [normalized_shape] if isinstance(
        normalized_shape, int) else normalized_shape

    dim = tuple([-i - 1 for i in range(len(normalized_shape))])

    with precision("float32"):
        varx = pow(input.cast(trt.float32), 2.0)
        varx = varx.mean(dim, keepdim=True)
        denom = varx + eps
        denom = denom.sqrt()
        y = input.cast(trt.float32) / denom
        y = y.cast(input.dtype)
    if weight is not None:
        y = y * weight

    return y


def rms_norm_op(
        input: Tensor,
        plugin_dtype: str,
        normalized_shape: Union[int, Tuple[int]],
        weight: Optional[Tensor] = None,
        eps: float = 1e-05,
        custom_plugin_paths=None,
) -> Tensor:
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    if custom_plugin_paths is None:
        custom_plugin_paths = []
    # create plugin
    if len(custom_plugin_paths) > 0:
        plugin_creator = trt.get_plugin_registry().get_plugin_creator(
            'Rmsnorm', '1'
        )
    else:
        plugin_creator = trt.get_plugin_registry().get_plugin_creator(
            'Rmsnorm', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plugin_creator is not None

    eps = trt.PluginField(
        "eps",
        np.array(eps, dtype=np.float32),
        trt.PluginFieldType.FLOAT32
    )

    type_id = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(plugin_dtype))], np.int32),
        trt.PluginFieldType.INT32)
    plugin_filed_collections = trt.PluginFieldCollection(
        [eps, type_id]
    )
    rmsnorm_plugin = plugin_creator.create_plugin(
        "rmsnorm_quantization", plugin_filed_collections
    )

    if weight is None:
        weight = constant(
            np.ones(normalized_shape, dtype=str_dtype_to_np(plugin_dtype)))
    inputs = [input.trt_tensor, weight.trt_tensor]
    layer = default_trtnet().add_plugin_v2(inputs, rmsnorm_plugin)
    return _create_tensor(layer.get_output(0), layer)


def trt_dtype_to_str(dtype: trt.DataType) -> str:
    _str_to_trt_dtype_dict = dict(float16=trt.float16,
                                  float32=trt.float32,
                                  int32=trt.int32,
                                  int8=trt.int8,
                                  bool=trt.bool,
                                  bfloat16=trt.bfloat16,
                                  fp8=trt.fp8)
    trt_to_str_dtype_dict = {v: k for k, v in _str_to_trt_dtype_dict.items()}
    return trt_to_str_dtype_dict[dtype]


class RmsNorm(Module):
    """
    Copy from tensorrt_llm.functional, for reduce some warning;
    """

    def __init__(
            self,
            normalized_shape,
            eps=1e-06,
            elementwise_affine=True,
            dtype=None,
            custom_plugin_paths=None
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        self.dtype = dtype
        if custom_plugin_paths is None:
            custom_plugin_paths = []
        self.custom_plugin_paths = custom_plugin_paths
        if self.elementwise_affine:
            self.weight = Parameter(shape=self.normalized_shape, dtype=dtype)
        else:
            self.register_parameter('weight', None)

        self.eps = eps

    def forward(self, x):
        weight = None if self.weight is None else self.weight.value
        return rms_norm(x, self.normalized_shape, weight, self.eps)
        # return rms_norm_op(
        #     x,
        #     plugin_dtype=trt_dtype_to_str(self.dtype),
        #     normalized_shape=self.normalized_shape,
        #     weight=weight,
        #     eps=self.eps,
        #     custom_plugin_paths=self.custom_plugin_paths
        # )


class QWen2Attention(Module):
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            max_position_embeddings,
            num_kv_heads=None,
            num_layers=1,
            apply_query_key_layer_scaling=False,
            attention_mask_type=AttentionMaskType.causal,
            bias=True,
            dtype=None,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=10000.0,
            rotary_embedding_scaling=None,
            rotary_embedding_percentage=1.0,
            tp_group=None,
            tp_size=1,
            tp_rank=0,
            quant_mode: QuantMode = QuantMode(0),
            q_scaling=1.0,
            cross_attention=False,
            relative_attention=False,
            max_distance=0,
            num_buckets=0,
            dense_bias=None,
            dense_context_fmha=False,
    ):
        super().__init__()
        self.cross_attention = cross_attention
        # max_positions = config.max_position_embeddings
        # self.register_buffer(
        #     "bias",
        #     torch.tril(
        #         torch.ones((max_positions, max_positions), dtype=torch.bool)
        #     ).view(1, 1, max_positions, max_positions),
        #     persistent=False,
        # )
        # self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)
        self.hidden_size = hidden_size
        self.split_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # self.use_flash_attn = False
        self.scale_attn_weights = True

        self.projection_size = hidden_size
        self.hidden_size_per_attention_head = (
                self.projection_size // num_attention_heads
        )
        # copy from chatglm6b trt-llm
        self.attention_mask_type = attention_mask_type
        self.bias = bias
        self.attention_head_size = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads // tp_size
        self.num_attention_kv_heads = (
                                              num_kv_heads + tp_size - 1
                                      ) // tp_size if num_kv_heads is not None else self.num_attention_heads
        self.hidden_size = hidden_size // tp_size
        self.max_position_embeddings = max_position_embeddings

        self.num_layers = num_layers
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.q_scaling = q_scaling
        if self.apply_query_key_layer_scaling:
            self.norm_factor *= self.num_layers
            self.q_scaling *= self.num_layers

        self.position_embedding_type = position_embedding_type
        self.dense_context_fmha = dense_context_fmha
        self.relative_attention = relative_attention
        self.max_distance = max_distance

        self.rotary_embedding_base = rotary_embedding_base
        self.rotary_embedding_scale_type = RotaryScalingType.none
        self.rotary_embedding_scale = 1.0
        if rotary_embedding_scaling is not None:
            assert rotary_embedding_scaling["type"] in ["linear", "dynamic"]
            self.rotary_embedding_scale_type = RotaryScalingType.linear if \
            rotary_embedding_scaling[
                "type"] == "linear" else RotaryScalingType.dynamic
            self.rotary_embedding_scale = rotary_embedding_scaling["factor"]
            assert self.rotary_embedding_scale > 1.0
        self.rotary_embedding_dim = 0
        if self.position_embedding_type == PositionEmbeddingType.rope_gpt_neox:
            self.rotary_embedding_dim = int(self.attention_head_size *
                                            rotary_embedding_percentage)
            # TODO: Once we add RotaryEmbedding outside GPTAttention plugin,
            #       we need to set it up here

        self.dtype = dtype
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        if dense_bias is None:
            dense_bias = bias
        self.quant_mode = quant_mode
        self.use_int8_kv_cache = self.quant_mode.has_int8_kv_cache()
        if self.use_int8_kv_cache:
            self.kv_orig_quant_scale = Parameter(shape=(1,), dtype='float32')
            self.kv_quant_orig_scale = Parameter(shape=(1,), dtype='float32')
        else:
            self.register_parameter('kv_orig_quant_scale', None)
            self.register_parameter('kv_quant_orig_scale', None)
        # self.c_attn = nn.Linear(config.hidden_size, 3 * self.projection_size)

        # self.c_proj = nn.Linear(
        #     config.hidden_size, self.projection_size, bias=not config.no_bias
        # )
        self.use_fp8_qdq = self.quant_mode.has_fp8_qdq()
        if self.use_fp8_qdq:
            self.qkv = FP8Linear(hidden_size,
                                 hidden_size +
                                 (2 * tp_size * self.num_attention_kv_heads *
                                  self.attention_head_size),
                                 bias=True,
                                 dtype=dtype,
                                 tp_group=tp_group,
                                 tp_size=tp_size,
                                 gather_output=False)
            self.o_proj = FP8RowLinear(hidden_size,
                                      hidden_size,
                                      bias=dense_bias,
                                      dtype=dtype,
                                      tp_group=tp_group,
                                      tp_size=tp_size)
        else:
            self.qkv = ColumnLinear(hidden_size,
                                    hidden_size +
                                    (2 * tp_size * self.num_attention_kv_heads *
                                     self.attention_head_size),
                                    bias=True,
                                    dtype=dtype,
                                    tp_group=tp_group,
                                    tp_size=tp_size,
                                    gather_output=False)
            self.o_proj = RowLinear(hidden_size,
                                   hidden_size,
                                   bias=dense_bias,
                                   dtype=dtype,
                                   tp_group=tp_group,
                                   tp_size=tp_size)

        if relative_attention:
            self.rel_attn_table = Parameter(shape=(num_attention_heads //
                                                   tp_size, num_buckets),
                                            dtype=dtype)

    def forward(
            self,
            hidden_states: Tensor,
            # position_embedding,
            use_cache=False,
            kv_cache_params=None,
            attention_params=None,
    ):
        if not default_net().plugin_config.gpt_attention_plugin:
            raise ValueError(
                'QWen is only supported with GPTAttention plugin')

        assert isinstance(hidden_states, Tensor)
        qkv = self.qkv(hidden_states)
        kv_orig_quant_scale = self.kv_orig_quant_scale.value if self.use_int8_kv_cache else None
        kv_quant_orig_scale = self.kv_quant_orig_scale.value if self.use_int8_kv_cache else None

        # return outputs
        context, past_key_value = gpt_attention(
            qkv=qkv,
            past_key_value=kv_cache_params.get_first_past_key_value(),
            sequence_length=attention_params.sequence_length,
            host_past_key_value_lengths=kv_cache_params.host_past_key_value_lengths,
            host_max_attention_window_sizes=kv_cache_params.host_max_attention_window_sizes,
            host_sink_token_length=kv_cache_params.host_sink_token_length,
            context_lengths=attention_params.context_lengths,
            cache_indirection=kv_cache_params.cache_indirection,
            host_request_types=attention_params.host_request_types,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_attention_kv_heads,
            hidden_size_per_head=self.attention_head_size,
            q_scaling=self.q_scaling,
            rotary_embedding_dim=self.rotary_embedding_dim,
            # when we use it 0, we will not use rotary embedding in plugin
            rotary_embedding_base=self.rotary_embedding_base,
            rotary_embedding_scale_type=self.rotary_embedding_scale_type,
            rotary_embedding_max_positions=self.max_position_embeddings,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            kv_orig_quant_scale=kv_orig_quant_scale,
            kv_quant_orig_scale=kv_quant_orig_scale,
            kv_cache_quant_mode=QuantMode.from_description(
                use_int8_kv_cache=self.use_int8_kv_cache),
            kv_cache_block_pointers=kv_cache_params.get_first_kv_cache_block_pointers(),
            host_kv_cache_block_pointers=kv_cache_params.get_first_host_kv_cache_block_pointers(),
            max_context_length=attention_params.max_context_length,
            mask_type=self.attention_mask_type.value,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            host_context_lengths=attention_params.host_context_lengths,
            dense_context_fmha=self.dense_context_fmha,
            use_cache=use_cache,
        )
        context = self.o_proj(context)
        if use_cache:
            return (context, past_key_value)
        else:
            return context


class QWen2MLP(Module):
    def __init__(
            self,
            hidden_size,
            ffn_hidden_size,
            hidden_act,
            bias=True,
            dtype=None,
            tp_group=None,
            tp_size=1
    ):
        super().__init__()
        if hidden_act not in ACT2FN:
            raise ValueError(
                'unsupported activation function: {}'.format(hidden_act))
        self.gate_proj = ColumnLinear(
            hidden_size,
            ffn_hidden_size,
            bias=bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            gather_output=False
        )
        self.up_proj = ColumnLinear(
            hidden_size,
            ffn_hidden_size,
            bias=bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            gather_output=False
        )
        self.down_proj = RowLinear(
            ffn_hidden_size,
            hidden_size,
            bias=bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size
        )
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.hidden_act = hidden_act
        self.bias = bias
        self.dtype = dtype

    def forward(self, hidden_states):
        a1 = self.gate_proj(hidden_states)
        a2 = self.up_proj(hidden_states)
        # intermediate_parallel = a1 * F.silu(a2)
        intermediate_parallel = a2 * ACT2FN[self.hidden_act](a1)
        output = self.down_proj(intermediate_parallel)
        return output


class Qwen2DecoderLayer(Module):
    def __init__(
            self,
            layer_id,
            hidden_size,
            num_attention_heads,
            max_position_embeddings,
            num_layers,
            dtype=None,
            attention_mask_type=AttentionMaskType.causal,
            apply_query_key_layer_scaling=False,
            hidden_act='silu',
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_base=10000.0,
            rotary_scaling=None,
            quant_mode=QuantMode(0),
            mlp_hidden_size=None,
            bias=False,
            tp_group=None,
            tp_size=1,
            tp_rank=0,
            rms_norm_eps=1e-06,
            dense_context_fmha=False,
            custom_plugin_paths=None
    ):
        super().__init__()
        if custom_plugin_paths is None:
            custom_plugin_paths = []
        self.custom_plugin_paths = custom_plugin_paths
        self._layer_id = layer_id  # useful for debugging
        self.hidden_size = hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.bias = bias
        self.dense_context_fmha=dense_context_fmha
        self.hidden_act = hidden_act
        self.dtype = dtype
        self.attention_mask_type = attention_mask_type
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.num_layers = num_layers
        self.position_embedding_type = position_embedding_type
        self.rotary_embedding_base = rotary_base
        self.rotary_embedding_scaling = rotary_scaling

        self.input_layernorm = RmsNorm(
            normalized_shape=hidden_size,
            eps=rms_norm_eps,
            dtype=dtype,
            custom_plugin_paths=self.custom_plugin_paths
        )

        self.self_attn = QWen2Attention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            num_layers=self.num_layers,
            dtype=self.dtype,
            attention_mask_type=self.attention_mask_type,
            bias=bias,
            position_embedding_type=self.position_embedding_type,
            rotary_embedding_base=self.rotary_embedding_base,
            rotary_embedding_scaling=self.rotary_embedding_scaling,
            tp_group=self.tp_group,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            quant_mode=quant_mode,
            dense_bias=bias,
            dense_context_fmha=self.dense_context_fmha,
        )
        if not mlp_hidden_size:
            mlp_hidden_size = hidden_size * 4

        self.post_attention_layernorm = RmsNorm(
            normalized_shape=hidden_size,
            eps=rms_norm_eps,
            dtype=dtype,
            custom_plugin_paths=custom_plugin_paths
        )
        self.mlp = QWen2MLP(
            hidden_size=hidden_size,
            ffn_hidden_size=mlp_hidden_size,
            hidden_act=hidden_act,
            dtype=dtype,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size
        )

    def forward(
            self,
            hidden_states: Tensor,
            # position_embedding,
            use_cache=False,
            kv_cache_params=None,
            attention_params=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # self.register_network_output("ln_1", identity_op(hidden_states))
        attention_output = self.self_attn(
            hidden_states,
            # position_embedding,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )
        if use_cache:
            attention_output, presents = attention_output

        self.register_network_output(
            "attention_output",
            identity_op(attention_output)
        )
        hidden_states = residual + attention_output

        residual = hidden_states

        # self.register_network_output("ln_2_input", identity_op(hidden_states))
        hidden_states = self.post_attention_layernorm(hidden_states)
        # self.register_network_output("ln_2_output", identity_op(hidden_states))

        hidden_states = self.mlp(hidden_states)
        # self.register_network_output("mlp_output", identity_op(hidden_states))

        hidden_states = residual + hidden_states
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class QWen2Model(Module):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 seq_length,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 dtype,
                 mlp_hidden_size=None,
                 position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
                 bias=False,
                 rotary_base=10000.0,
                 rotary_scaling=None,
                 mapping=Mapping(),
                 quant_mode=QuantMode(0),
                 use_parallel_embedding=False,
                 embedding_sharding_dim=0,
                 rms_norm_eps=1e-06,
                 dense_context_fmha=False,
                 use_prompt_tuning=False,
                 custom_plugin_paths=None,
                 ):
        super().__init__()
        self.mapping = mapping
        if custom_plugin_paths is None:
            custom_plugin_paths = []
        if self.mapping.is_first_pp_rank():
            # self.embed_tokens = Embedding(
            #     num_embeddings=vocab_size,
            #     embedding_dim=hidden_size,
            #     dtype=dtype,
            #     tp_size=mapping.tp_size if use_parallel_embedding else 1,
            #     tp_group=mapping.tp_group if use_parallel_embedding else None,
            #     sharding_dim=embedding_sharding_dim,
            #     tp_rank=mapping.tp_rank
            # )
            self.embed_tokens = GPTEmbedding(
                vocab_size,
                hidden_size,
                max_position_embeddings,
                position_embedding_type=PositionEmbeddingType.relative,
                dtype=dtype,
                use_prompt_tuning=use_prompt_tuning,
                tensor_parallel=mapping.tp_size
                if use_parallel_embedding else 1,
                tensor_parallel_group=mapping.tp_group
                if use_parallel_embedding else None,
                sharding_dim=embedding_sharding_dim,
                tp_rank=mapping.tp_rank
            )

        self.layers = ModuleList([
            Qwen2DecoderLayer(
                layer_id=i,
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                num_layers=num_layers,
                max_position_embeddings=max_position_embeddings,
                dtype=dtype,
                hidden_act=hidden_act,
                quant_mode=quant_mode,
                mlp_hidden_size=mlp_hidden_size,
                position_embedding_type=position_embedding_type,
                rotary_base=rotary_base,
                rotary_scaling=rotary_scaling,
                bias=bias,
                tp_group=mapping.tp_group,
                tp_size=mapping.tp_size,
                tp_rank=mapping.tp_rank,
                rms_norm_eps=rms_norm_eps,
                dense_context_fmha=dense_context_fmha,
                custom_plugin_paths=custom_plugin_paths
            )
            for i in self.mapping.pp_layers(num_layers)
        ])

        self.norm = RmsNorm(
            normalized_shape=hidden_size,
            eps=rms_norm_eps,
            dtype=dtype,
        )

    def forward(
            self,
            input_ids,
            position_ids=None,
            use_cache=False,
            kv_cache_params=None,
            attention_params=None,
            hidden_states=None,
            prompt_embedding_table=None,
            prompt_tasks=None,
            prompt_vocab_size=None
    ):

        if kv_cache_params.past_key_value is None:
            tuple([None] * len(self.layers))

        if use_cache:
            presents = []

        if self.mapping.is_first_pp_rank():
            # hidden_states = self.embed_tokens(input_ids)
            hidden_states = self.embed_tokens(input_ids, position_ids,
                                           prompt_embedding_table, prompt_tasks,
                                           prompt_vocab_size)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())
        self.register_network_output(f"embd", hidden_states)

        for layer, past, pointer, host_pointer, max_attention_window_size in zip(
                self.layers,
                kv_cache_params.past_key_value,
                kv_cache_params.kv_cache_block_pointers,
                kv_cache_params.host_kv_cache_block_pointers,
                kv_cache_params.host_max_attention_window_sizes
        ):
            hidden_states = layer(
                hidden_states,
                use_cache=use_cache,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.host_past_key_value_lengths,
                    host_max_attention_window_sizes=max_attention_window_size,
                    host_sink_token_length=kv_cache_params.host_sink_token_length,
                    kv_cache_block_pointers=[pointer],
                    host_kv_cache_block_pointers=[host_pointer],
                    cache_indirection=kv_cache_params.cache_indirection),
                attention_params=attention_params,
            )

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        if self.mapping.is_last_pp_rank():
            hidden_states = self.norm(hidden_states)
            # self.register_network_output("norm_output",
            #                              identity_op(hidden_states))
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class Qwen2ForCausalLM(QWen2Model, GenerationMixin):

    def __init__(self,
                 num_layers,
                 num_heads,
                 num_kv_heads,
                 hidden_size,
                 seq_length,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 dtype,
                 logits_dtype="float32",
                 mlp_hidden_size=None,
                 position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
                 rotary_base=10000.0,
                 rotary_scaling=None,
                 mapping=Mapping(),
                 quant_mode=QuantMode(0),
                 use_parallel_embedding=False,
                 embedding_sharding_dim=0,
                 rms_norm_eps=1e-06,
                 dense_context_fmha=False,
                 use_prompt_tuning=False,
                 custom_plugin_paths=None,
                 ):
        self.mapping = mapping
        if isinstance(dtype, str):
            self.dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self.dtype = dtype
        if isinstance(logits_dtype, str):
            self.logits_dtype = str_dtype_to_trt(logits_dtype)
        else:
            assert isinstance(logits_dtype, trt.DataType)
            self.logits_dtype = logits_dtype
        self.num_layers = num_layers
        self.num_heads = num_heads
        if num_kv_heads is None or num_kv_heads <= 0:
            num_kv_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tp_size = mapping.tp_size

        self.kv_dtype = self.dtype
        if quant_mode.has_int8_kv_cache():
            self.kv_dtype = str_dtype_to_trt('int8')
        elif quant_mode.has_fp8_kv_cache():
            self.kv_dtype = str_dtype_to_trt('fp8')
        self.quant_mode = quant_mode
        self.use_parallel_embedding = use_parallel_embedding
        self.embedding_sharding_dim = embedding_sharding_dim

        # set custom plugin path
        if custom_plugin_paths is None:
            custom_plugin_paths = []
        self.custom_plugin_paths = custom_plugin_paths

        super().__init__(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
            seq_length=seq_length,
            vocab_size=vocab_size,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            dtype=dtype,
            mlp_hidden_size=mlp_hidden_size,
            position_embedding_type=position_embedding_type,
            rotary_base=rotary_base,
            rotary_scaling=rotary_scaling,
            mapping=mapping,
            quant_mode=quant_mode,
            use_parallel_embedding=use_parallel_embedding,
            embedding_sharding_dim=embedding_sharding_dim,
            rms_norm_eps=rms_norm_eps,
            dense_context_fmha=dense_context_fmha,
            use_prompt_tuning=use_prompt_tuning,
            custom_plugin_paths=custom_plugin_paths
        )
        vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
        if self.mapping.is_last_pp_rank():
            self.lm_head = ColumnLinear(hidden_size,
                                        vocab_size_padded,
                                        bias=False,
                                        dtype=dtype,
                                        tp_group=mapping.tp_group,
                                        tp_size=mapping.tp_size,
                                        gather_output=True)

    def forward(
            self,
            input_ids,
            position_ids=None,
            use_cache=False,
            last_token_ids=None,
            kv_cache_params=None,
            attention_params=None,
            hidden_states=None,
            prompt_embedding_table=None,
            prompt_tasks=None,
            prompt_vocab_size=None
    ):
        hidden_states = super().forward(
            input_ids,
            position_ids,
            use_cache,
            kv_cache_params,
            attention_params,
            hidden_states,
            prompt_embedding_table=prompt_embedding_table,
            prompt_tasks=prompt_tasks,
            prompt_vocab_size=prompt_vocab_size,
        )
        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = gather_last_token_logits(
                hidden_states, last_token_ids,
                default_net().plugin_config.remove_input_padding)

            # [batch_size, hidden_size] -> [batch_size, vocab_size]
            lm_logits = self.lm_head(hidden_states)
            lm_logits.mark_output('logits', self.logits_dtype)
        else:
            hidden_states.mark_output('hidden_states_output', self.dtype)

        if use_cache and default_net().plugin_config.paged_kv_cache == False:
            for i, present in zip(self.mapping.pp_layers(self.num_layers),
                                  presents):
                present.mark_output(f'present_key_value_{i}', self.kv_dtype)
            if self.mapping.is_last_pp_rank():
                return (lm_logits, presents)
            return (hidden_states, presents)
        else:
            if self.mapping.is_last_pp_rank():
                return lm_logits
            return hidden_states

    def prepare_inputs(
            self,
            max_batch_size,
            max_input_len,
            max_output_len,
            use_cache,
            max_beam_width: int = 1,
            max_num_tokens: int = None,
            prompt_embedding_table_size=256,
    ):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        head_size = self.hidden_size // self.num_heads
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net().plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        tokens_per_block = default_net().plugin_config.tokens_per_block
        use_custom_all_reduce = default_net().plugin_config.use_custom_all_reduce

        model_inputs = self.prepare_basic_inputs(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_seq_len=max_output_len,
            num_kv_heads=self.num_kv_heads,
            head_size=head_size,
            num_layers=self.num_layers,
            kv_dtype=self.kv_dtype,
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            use_custom_all_reduce=use_custom_all_reduce,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            dtype=self.dtype,
            num_heads=self.num_heads,
            mapping=self.mapping,
            max_num_tokens=max_num_tokens,
            prompt_embedding_table_size=prompt_embedding_table_size,
        )

        return (model_inputs['input_ids'], model_inputs['position_ids'], True,
                model_inputs['last_token_ids'],
                KeyValueCacheParams(
                    past_key_value=model_inputs['past_key_value'],
                    host_past_key_value_lengths=model_inputs[
                        'host_past_key_value_lengths'],
                    host_max_attention_window_sizes=model_inputs[
                        'host_max_attention_window_sizes'],
                    host_sink_token_length=model_inputs['host_sink_token_length'],
                    kv_cache_block_pointers=model_inputs[
                        'kv_cache_block_pointers_list'],
                    host_kv_cache_block_pointers=model_inputs[
                        'host_kv_cache_block_pointers_list'],
                    cache_indirection=model_inputs['cache_indirection'],
                ),
                AttentionParams(
                    sequence_length=model_inputs['sequence_length'],
                    context_lengths=model_inputs['context_lengths'],
                    host_context_lengths=model_inputs['host_context_lengths'],
                    max_context_length=max_input_len,
                    host_request_types=model_inputs['host_request_types']),
                model_inputs['hidden_states_input'],
                model_inputs['prompt_embedding_table'],
                model_inputs['tasks'],
                model_inputs['prompt_vocab_size'])
