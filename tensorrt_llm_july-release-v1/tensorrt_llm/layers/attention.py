import enum
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .._common import default_net, precision
from ..functional import (RaggedTensor, Tensor, cast, clip, concat, constant,
                          generate_alibi_biases, generate_alibi_slopes,
                          gpt_attention, inflight_batching_gpt_attention,
                          matmul, round, shape, slice, softmax, split)
from ..module import Module
from ..parameter import Parameter
from .linear import ColumnLinear, RowLinear


class AttentionMaskType(enum.Enum):
    padding = 0
    causal = 1
    bidirectional = 2


class PositionEmbeddingType(enum.Enum):
    learned_absolute = enum.auto()
    rope = enum.auto()
    alibi = enum.auto()


@dataclass
class InflightBatchingParam:
    host_beam_widths: Tensor
    cache_indir_pointers: Tensor
    host_req_cache_max_seq_lengths: Tensor
    host_input_lengths: Tensor
    past_key_value_pointers: Tensor
    max_input_length: int
    max_beam_width: int
    kv_orig_quant_scale: Optional[Tensor] = None
    kv_quant_orig_scale: Optional[Tensor] = None
    use_int8_kv_cache: bool = False

    def __post_init__(self):
        assert self.max_input_length > 0, f"max_input_length must be positive, got {self.max_input_length}"
        assert self.max_beam_width > 0, f"max_beam_width must be positive, got {self.max_beam_width}"


class Attention(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 num_layers=1,
                 apply_query_key_layer_scaling=False,
                 attention_mask_type=AttentionMaskType.padding,
                 bias=True,
                 dtype=None,
                 position_embedding_type=PositionEmbeddingType.learned_absolute,
                 neox_rotary_style=False,
                 use_int8_kv_cache=False,
                 rotary_embedding_percentage=1.0,
                 tp_group=None,
                 tp_size=1,
                 multi_block_mode=False,
                 multi_query_mode=False):
        super().__init__()

        self.attention_mask_type = attention_mask_type
        self.attention_head_size = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads // tp_size
        self.num_attention_kv_heads = 1 if multi_query_mode else self.num_attention_heads
        self.hidden_size = hidden_size // tp_size
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

        self.rotary_embedding_dim = 0
        self.neox_rotary_style = neox_rotary_style
        if self.position_embedding_type == PositionEmbeddingType.rope:
            self.rotary_embedding_dim = int(self.attention_head_size *
                                            rotary_embedding_percentage)
            # TODO: Once we add RotaryEmbedding outside GPTAttention plugin,
            #       we need to set it up here

        self.dtype = dtype

        self.use_int8_kv_cache = use_int8_kv_cache
        if self.use_int8_kv_cache:
            self.kv_orig_quant_scale = Parameter(shape=(1, ), dtype='float32')
            self.kv_quant_orig_scale = Parameter(shape=(1, ), dtype='float32')
        else:
            self.register_parameter('kv_orig_quant_scale', None)
            self.register_parameter('kv_quant_orig_scale', None)

        # Note: in multi_query_mode, only query heads are split between multiple GPUs,
        # while key/value head are not split as there is only one head per key/value.
        # The output feature size is therefore (h/tp + 2) * d, where h is num_heads,
        # d is head_size, and tp is tensor_parallel_size.
        # In ColumnLinear op, the output dim is calculated by (h + 2*tp) * d / tp,
        # which matches the desired output size (h/tp + 2) * d after splitting
        self.qkv = ColumnLinear(hidden_size,
                                hidden_size *
                                3 if not multi_query_mode else hidden_size +
                                2 * tp_size * self.attention_head_size,
                                bias=bias,
                                dtype=dtype,
                                tp_group=tp_group,
                                tp_size=tp_size,
                                gather_output=False)
        self.dense = RowLinear(hidden_size,
                               hidden_size,
                               bias=bias,
                               dtype=dtype,
                               tp_group=tp_group,
                               tp_size=tp_size)

    def forward(self,
                hidden_states: RaggedTensor,
                attention_mask=None,
                past_key_value=None,
                sequence_length=None,
                past_key_value_length=None,
                masked_tokens=None,
                use_cache=False,
                cache_indirection=None,
                kv_cache_block_pointers=None,
                inflight_batching_args: Optional[InflightBatchingParam] = None,
                past_key_value_pointers=None):

        if self.position_embedding_type == PositionEmbeddingType.rope:
            if not default_net().plugin_config.gpt_attention_plugin:
                raise ValueError(
                    'RoPE is only supported with GPTAttention plugin')
        assert isinstance(hidden_states, RaggedTensor)
        input_lengths = hidden_states.row_lengths
        max_input_length = hidden_states.max_row_length
        hidden_states = hidden_states.data
        qkv = self.qkv(hidden_states)

        if default_net().plugin_config.gpt_attention_plugin:
            assert sequence_length is not None
            assert past_key_value_length is not None
            assert masked_tokens is not None
            assert cache_indirection is not None
            assert self.attention_mask_type in [
                AttentionMaskType.causal, AttentionMaskType.bidirectional
            ], 'Plugin only support masked MHA.'
            assert input_lengths is not None
            kv_orig_quant_scale = self.kv_orig_quant_scale.value if self.use_int8_kv_cache else None
            kv_quant_orig_scale = self.kv_quant_orig_scale.value if self.use_int8_kv_cache else None
            if self.position_embedding_type == PositionEmbeddingType.alibi:
                raise ValueError(
                    'ALiBi is only supported without GPTAttention plugin')
            context, past_key_value = gpt_attention(
                qkv,
                past_key_value,
                sequence_length,
                past_key_value_length,
                masked_tokens,
                input_lengths,
                max_input_length,
                cache_indirection,
                self.num_attention_heads,
                self.attention_head_size,
                self.q_scaling,
                self.rotary_embedding_dim,
                self.neox_rotary_style,
                self.multi_block_mode,
                self.multi_query_mode,
                kv_orig_quant_scale,
                kv_quant_orig_scale,
                self.use_int8_kv_cache,
                kv_cache_block_pointers=kv_cache_block_pointers)

        elif default_net().plugin_config.inflight_batching_gpt_attention_plugin:
            # The inflight batching mode
            assert inflight_batching_args is not None
            context, past_key_value = inflight_batching_gpt_attention(
                qkv,
                host_beam_widths=inflight_batching_args.host_beam_widths,
                host_input_lengths=inflight_batching_args.host_input_lengths,
                input_lengths=input_lengths,
                past_key_value=past_key_value,
                past_key_value_pointers=past_key_value_pointers,
                host_past_key_value_lengths=past_key_value_length,
                cache_indirection_pointers=inflight_batching_args.
                cache_indir_pointers,
                host_req_cache_max_seq_lengths=inflight_batching_args.
                host_req_cache_max_seq_lengths,
                num_heads=self.num_attention_heads,
                head_size=self.attention_head_size,
                q_scaling=self.q_scaling,
                rotary_embedding_dim=self.rotary_embedding_dim,
                neox_rotary_style=self.neox_rotary_style,
                multi_block_mode=self.multi_block_mode,
                multi_query_mode=self.multi_query_mode,
                kv_orig_quant_scale=inflight_batching_args.kv_orig_quant_scale,
                kv_quant_orig_scale=inflight_batching_args.kv_quant_orig_scale,
                use_int8_kv_cache=inflight_batching_args.use_int8_kv_cache,
                max_input_len=inflight_batching_args.max_input_length,
                max_beam_width=inflight_batching_args.max_beam_width,
                kv_cache_block_pointers=kv_cache_block_pointers,
                pointers_to_kv_cache_block_pointers=None)

        else:
            assert default_net().plugin_config.paged_kv_cache == False

            def transpose_for_scores(x, is_kv: bool = False):
                _num_attention_heads = self.num_attention_kv_heads if is_kv else self.num_attention_heads
                new_x_shape = concat([
                    shape(x, 0),
                    shape(x, 1), _num_attention_heads, self.attention_head_size
                ])
                return x.view(new_x_shape).permute([0, 2, 1, 3])

            # When self.multi_query_mode == True, qkv after projection is of shape
            #   [bs, seqlen, (num_attention_heads + 2) * attention_head_size]
            # instead of
            #   [bs, seqlen, 3, num_attention_heads, attention_head_size].
            # The projected and split qkv after transpose_for_scores():
            #   Q[bs, num_attention_heads, seqlen, attention_head_size]
            #   K[bs, 1, seqlen, attention_head_size]
            #   V[bs, 1, seqlen, attention_head_size]
            if self.multi_query_mode:
                query, key, value = split(qkv, [
                    self.hidden_size, self.attention_head_size,
                    self.attention_head_size
                ],
                                          dim=2)
            else:
                query, key, value = split(qkv, self.hidden_size, dim=2)
            query = transpose_for_scores(query)
            key = transpose_for_scores(key, is_kv=True)
            value = transpose_for_scores(value, is_kv=True)

            if past_key_value is not None:

                def dequantize_tensor(x, scale):
                    # Cast from int8 to dtype
                    casted_x = cast(x, self.dtype)
                    return casted_x * scale

                if self.use_int8_kv_cache:
                    past_key_value = dequantize_tensor(
                        past_key_value, self.kv_quant_orig_scale.value)

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
                # FIXME(kaiyu): Remove cast after https://nvbugs/4211574 is fixed
                key = concat([past_key, key], dim=2).cast(self.dtype)
                value = concat([past_value, value], dim=2).cast(self.dtype)

            if use_cache:
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

                if self.use_int8_kv_cache:

                    def quantize_tensor(x, scale):
                        scaled = x * scale
                        rounded = round(scaled)
                        clipped = clip(rounded, -128, 127)
                        quantized = cast(clipped, 'int8')
                        return quantized

                    past_key_value = quantize_tensor(
                        past_key_value, self.kv_orig_quant_scale.value)

            key_length = shape(key, 2)

            # The following code creates a 2D tensor with 0s in the lower triangular (including the diagonal) and
            # +INF in the upper triangular parts. This bias tensor will be added to the output of the Q*K^T matrix
            # multiplication (BMM1). The +INF elements will be transformed to 0s by the Softmax operator that
            # follows. The elements that corresponds to 0s in the bias are unaffected by the bias tensor.
            #
            # Note that when we added to another bias tensor B (for example, with AliBi), the values in the lower-
            # triangular part of the B tensor are not affected and the upper-triangular ones are set to +INF.
            if self.attention_mask_type == AttentionMaskType.causal:
                query_length = shape(query, 2)
                starts = concat([0, 0, key_length - query_length, 0])
                sizes = concat([1, 1, query_length, key_length])
                select_buf = np.expand_dims(
                    np.tril(
                        np.ones((self.max_position_embeddings,
                                 self.max_position_embeddings))).astype(bool),
                    (0, 1))

                select_buf = np.logical_not(select_buf)
                mask_buf = np.zeros_like(select_buf, np.float32)
                mask_buf[select_buf] = float('-inf')
                buffer = constant(mask_buf)
                causal_mask = slice(buffer, starts, sizes)

            bias = attention_mask
            if self.position_embedding_type == PositionEmbeddingType.alibi:
                alibi_slopes = generate_alibi_slopes(self.num_attention_heads)
                alibi_biases = generate_alibi_biases(alibi_slopes, key_length)
                bias = alibi_biases if bias is None else bias + alibi_biases

            key = key.permute([0, 1, 3, 2])
            with precision('float32'):
                attention_scores = matmul(cast(query, 'float32'),
                                          cast(key, 'float32'))

                attention_scores = attention_scores / self.norm_factor

                if self.attention_mask_type == AttentionMaskType.causal:
                    bias = causal_mask if bias is None else bias + causal_mask

                if bias is not None:
                    attention_scores = attention_scores + bias

                attention_probs = softmax(attention_scores, dim=-1)

            context = matmul(attention_probs, value).permute([0, 2, 1, 3])
            context = context.view(
                concat([shape(context, 0),
                        shape(context, 1), self.hidden_size]))

        context = self.dense(context)

        context = RaggedTensor.from_row_lengths(context, input_lengths,
                                                max_input_length)
        if use_cache:
            return (context, past_key_value)
        else:
            return context
