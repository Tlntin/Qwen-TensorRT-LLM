from collections import OrderedDict

import tensorrt as trt

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import (RaggedTensor, Tensor, assertion,
                           gather_last_token_logits, gpt_attention,
                           inflight_batching_gpt_attention, shape)
from ...layers import (MLP, AttentionMaskType, ColumnLinear, Embedding,
                       InflightBatchingParam, LayerNorm, RowLinear)
from ...module import Module, ModuleList
from ...parameter import Parameter
from ...quantization import FP8MLP, FP8Linear, FP8RowLinear, QuantMode


class GPTJAttention(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 rotary_dim,
                 max_position_embeddings,
                 dtype=None,
                 multi_block_mode=False,
                 quant_mode=QuantMode(0)):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rotary_dim = rotary_dim
        self.neox_rotary_style = False
        self.multi_block_mode = multi_block_mode
        self.multi_query_mode = False

        self.dtype = dtype

        self.use_int8_kv_cache = quant_mode.has_int8_kv_cache()
        self.use_fp8_kv_cache = quant_mode.has_fp8_kv_cache()
        if self.use_int8_kv_cache or self.use_fp8_kv_cache:
            self.kv_quantization_scale = Parameter(shape=(1, ), dtype='float32')
            self.kv_dequantization_scale = Parameter(shape=(1, ),
                                                     dtype='float32')
        else:
            self.register_parameter('kv_quantization_scale', None)
            self.register_parameter('kv_dequantization_scale', None)

        self.qkv = ColumnLinear(hidden_size,
                                hidden_size * 3,
                                bias=False,
                                gather_output=False,
                                dtype=dtype)
        self.dense = RowLinear(hidden_size,
                               hidden_size,
                               bias=False,
                               dtype=dtype)

    def forward(self,
                hidden_states: RaggedTensor,
                attention_mask=None,
                past_key_value=None,
                sequence_length=None,
                past_key_value_length=None,
                masked_tokens=None,
                use_cache=False,
                cache_indirection=None,
                inflight_batching_args=None,
                past_key_value_pointers=None):
        input_lengths = hidden_states.row_lengths
        max_input_length = hidden_states.max_row_length
        if not default_net(
        ).plugin_config.gpt_attention_plugin and not default_net(
        ).plugin_config.inflight_batching_gpt_attention_plugin:
            raise ValueError(
                'GPT-J RoPE is only supported with GPTAttention and ibGPTAttention plugin'
            )
        qkv = self.qkv(hidden_states.data)
        assert past_key_value_length is not None
        kv_quantization_scale = self.kv_quantization_scale.value if (
            self.use_int8_kv_cache or self.use_fp8_kv_cache) else None
        kv_dequantization_scale = self.kv_dequantization_scale.value if (
            self.use_int8_kv_cache or self.use_fp8_kv_cache) else None
        if default_net().plugin_config.gpt_attention_plugin:
            assert sequence_length is not None
            assert masked_tokens is not None
            assert cache_indirection is not None
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
                1.0,
                self.rotary_dim,
                self.neox_rotary_style,
                self.multi_block_mode,
                self.multi_query_mode,
                kv_quantization_scale,
                kv_dequantization_scale,
                self.use_int8_kv_cache,
                use_fp8_kv_cache=self.use_fp8_kv_cache)
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
                q_scaling=1.0,
                rotary_embedding_dim=self.rotary_dim,
                neox_rotary_style=self.neox_rotary_style,
                multi_block_mode=self.multi_block_mode,
                multi_query_mode=self.multi_query_mode,
                kv_orig_quant_scale=kv_quantization_scale,
                kv_quant_orig_scale=kv_dequantization_scale,
                use_int8_kv_cache=inflight_batching_args.use_int8_kv_cache,
                max_input_len=inflight_batching_args.max_input_length,
                max_beam_width=inflight_batching_args.max_beam_width,
                kv_cache_block_pointers=None,
                pointers_to_kv_cache_block_pointers=None,
                use_fp8_kv_cache=self.use_fp8_kv_cache)

        context = self.dense(context)
        context = RaggedTensor.from_row_lengths(context, input_lengths,
                                                max_input_length)

        if use_cache:
            return (context, past_key_value)

        return context


class FP8GPTJAttention(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 rotary_dim,
                 max_position_embeddings,
                 dtype=None,
                 multi_block_mode=False,
                 quant_mode=QuantMode(0)):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rotary_dim = rotary_dim
        self.neox_rotary_style = False
        self.multi_block_mode = multi_block_mode
        self.multi_query_mode = False

        self.dtype = dtype

        self.use_int8_kv_cache = quant_mode.has_int8_kv_cache()
        self.use_fp8_kv_cache = quant_mode.has_fp8_kv_cache()
        if self.use_int8_kv_cache or self.use_fp8_kv_cache:
            self.kv_quantization_scale = Parameter(shape=(1, ), dtype='float32')
            self.kv_dequantization_scale = Parameter(shape=(1, ),
                                                     dtype='float32')
        else:
            self.register_parameter('kv_quantization_scale', None)
            self.register_parameter('kv_dequantization_scale', None)

        self.qkv = FP8Linear(hidden_size,
                             hidden_size * 3,
                             bias=False,
                             gather_output=False,
                             dtype=dtype)
        self.dense = FP8RowLinear(hidden_size,
                                  hidden_size,
                                  bias=False,
                                  dtype=dtype)

    def forward(self,
                hidden_states: RaggedTensor,
                attention_mask=None,
                past_key_value=None,
                sequence_length=None,
                past_key_value_length=None,
                masked_tokens=None,
                use_cache=False,
                cache_indirection=None,
                inflight_batching_args=None,
                past_key_value_pointers=None):
        input_lengths = hidden_states.row_lengths
        max_input_length = hidden_states.max_row_length
        if not (default_net().plugin_config.gpt_attention_plugin or default_net(
        ).plugin_config.inflight_batching_gpt_attention_plugin):
            raise ValueError(
                'GPT-J RoPE is only supported with GPTAttention plugin')
        qkv = self.qkv(hidden_states.data)
        assert past_key_value_length is not None
        kv_quantization_scale = self.kv_quantization_scale.value if (
            self.use_int8_kv_cache or self.use_fp8_kv_cache) else None
        kv_dequantization_scale = self.kv_dequantization_scale.value if (
            self.use_int8_kv_cache or self.use_fp8_kv_cache) else None
        if default_net().plugin_config.gpt_attention_plugin:
            assert sequence_length is not None
            assert masked_tokens is not None
            assert cache_indirection is not None
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
                1.0,
                self.rotary_dim,
                self.neox_rotary_style,
                self.multi_block_mode,
                self.multi_query_mode,
                kv_quantization_scale,
                kv_dequantization_scale,
                self.use_int8_kv_cache,
                use_fp8_kv_cache=self.use_fp8_kv_cache)

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
                q_scaling=1.0,
                rotary_embedding_dim=self.rotary_dim,
                neox_rotary_style=self.neox_rotary_style,
                multi_block_mode=self.multi_block_mode,
                multi_query_mode=self.multi_query_mode,
                kv_orig_quant_scale=kv_quantization_scale,
                kv_quant_orig_scale=kv_dequantization_scale,
                use_int8_kv_cache=inflight_batching_args.use_int8_kv_cache,
                max_input_len=inflight_batching_args.max_input_length,
                max_beam_width=inflight_batching_args.max_beam_width,
                kv_cache_block_pointers=None,
                pointers_to_kv_cache_block_pointers=None,
                use_fp8_kv_cache=self.use_fp8_kv_cache)

        context = self.dense(context)
        context = RaggedTensor.from_row_lengths(context, input_lengths,
                                                max_input_length)

        if use_cache:
            return (context, past_key_value)

        return context


class GPTJDecoderLayer(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 num_layers,
                 rotary_dim,
                 dtype=None,
                 apply_query_key_layer_scaling=False,
                 attention_mask_type=AttentionMaskType.causal,
                 hidden_act='relu',
                 tp_group=None,
                 tp_size=1,
                 fp8_mode=False,
                 quant_mode=QuantMode(0)):
        super().__init__()
        self.input_layernorm = LayerNorm(normalized_shape=hidden_size,
                                         dtype=dtype)

        if fp8_mode:
            self.attention = FP8GPTJAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position_embeddings,
                dtype=dtype,
                quant_mode=quant_mode)

            self.mlp = FP8MLP(hidden_size=hidden_size,
                              ffn_hidden_size=hidden_size * 4,
                              hidden_act=hidden_act,
                              dtype=dtype,
                              tp_group=tp_group,
                              tp_size=tp_size)
        else:
            self.attention = GPTJAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position_embeddings,
                dtype=dtype)

            self.mlp = MLP(hidden_size=hidden_size,
                           ffn_hidden_size=hidden_size * 4,
                           hidden_act=hidden_act,
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
                inflight_batching_args=None,
                past_key_value_pointers=None):
        if not default_net(
        ).plugin_config.layernorm_plugin and trt.__version__[:3] == '8.6':
            raise AssertionError(
                "You need to enable the LayerNorm plugin for GPT-J with TensorRT 8.6. Please set plugin_config.layernorm_plugin"
            )
        input_lengths = hidden_states.row_lengths
        max_input_length = hidden_states.max_row_length
        residual = hidden_states.data

        hidden_states = self.input_layernorm(hidden_states.data)

        attention_output = self.attention(
            RaggedTensor.from_row_lengths(hidden_states, input_lengths,
                                          max_input_length),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            sequence_length=sequence_length,
            past_key_value_length=past_key_value_length,
            masked_tokens=masked_tokens,
            use_cache=use_cache,
            cache_indirection=cache_indirection,
            inflight_batching_args=inflight_batching_args,
            past_key_value_pointers=past_key_value_pointers)

        if use_cache:
            attention_output, presents = attention_output
        attention_output = attention_output.data

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attention_output + feed_forward_hidden_states + residual
        hidden_states = RaggedTensor.from_row_lengths(hidden_states,
                                                      input_lengths,
                                                      max_input_length)
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class GPTJModel(Module):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 rotary_dim,
                 dtype=None,
                 tensor_parallel=1,
                 tensor_parallel_group=None,
                 apply_query_key_layer_scaling=False,
                 fp8_mode=False,
                 quant_mode=QuantMode(0)):
        super().__init__()
        self.embedding = Embedding(vocab_size, hidden_size, dtype=dtype)

        self.layers = ModuleList([
            GPTJDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                max_position_embeddings=max_position_embeddings,
                num_layers=num_layers,
                rotary_dim=rotary_dim,
                dtype=dtype,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                attention_mask_type=AttentionMaskType.causal,
                hidden_act=hidden_act,
                tp_group=tensor_parallel_group,
                tp_size=tensor_parallel,
                fp8_mode=fp8_mode,
                quant_mode=quant_mode) for _ in range(num_layers)
        ])

        self.ln_f = LayerNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(self,
                input_ids=None,
                position_ids=None,
                past_key_value=None,
                sequence_length=None,
                past_key_value_length=None,
                masked_tokens=None,
                use_cache=False,
                cache_indirection=None,
                inflight_batching_args=None):
        hidden_states = self.embedding(input_ids.data)

        if past_key_value is None:
            past_key_value = tuple([None] * len(self.layers))

        if use_cache:
            presents = []

        hidden_states = RaggedTensor.from_row_lengths(hidden_states,
                                                      input_ids.row_lengths,
                                                      input_ids.max_row_length)
        for idx, (layer, past) in enumerate(zip(self.layers, past_key_value)):
            hidden_states = layer(
                hidden_states,
                past_key_value=past,
                sequence_length=sequence_length,
                past_key_value_length=past_key_value_length,
                masked_tokens=masked_tokens,
                use_cache=use_cache,
                cache_indirection=cache_indirection,
                inflight_batching_args=inflight_batching_args,
                past_key_value_pointers=None if inflight_batching_args is None
                else inflight_batching_args.past_key_value_pointers[idx])

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        hidden_states = self.ln_f(hidden_states.data)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class GPTJForCausalLM(GPTJModel):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 rotary_dim,
                 dtype,
                 tensor_parallel=1,
                 tensor_parallel_group=None,
                 apply_query_key_layer_scaling=False,
                 fp8_mode=False,
                 quant_mode=QuantMode(0)):
        if isinstance(dtype, str):
            self._dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self._dtype = dtype
        self._kv_dtype = dtype
        self.quant_mode = quant_mode
        if quant_mode.has_int8_kv_cache():
            self._kv_dtype = str_dtype_to_trt('int8')
        # elif quant_mode.has_fp8_kv_cache():
        #     self._kv_dtype = str_dtype_to_trt('fp8')

        self._num_layers = num_layers
        self._num_heads = num_heads
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._tensor_parallel = tensor_parallel
        super().__init__(num_layers, num_heads, hidden_size, vocab_size,
                         hidden_act, max_position_embeddings, rotary_dim, dtype,
                         tensor_parallel, tensor_parallel_group,
                         apply_query_key_layer_scaling, fp8_mode, quant_mode)
        vocab_size_padded = pad_vocab_size(vocab_size, tensor_parallel)
        self.lm_head = ColumnLinear(hidden_size,
                                    vocab_size_padded,
                                    bias=True,
                                    dtype=dtype,
                                    tp_group=tensor_parallel_group,
                                    tp_size=tensor_parallel,
                                    gather_output=True)

    def forward(self,
                input_ids=None,
                position_ids=None,
                past_key_value=None,
                sequence_length=None,
                past_key_value_length=None,
                masked_tokens=None,
                use_cache=False,
                last_token_ids=None,
                cache_indirection=None,
                inflight_batching_args=None):
        hidden_states = super().forward(input_ids, position_ids, past_key_value,
                                        sequence_length, past_key_value_length,
                                        masked_tokens, use_cache,
                                        cache_indirection,
                                        inflight_batching_args)

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = gather_last_token_logits(
            hidden_states, last_token_ids,
            default_net().plugin_config.remove_input_padding)

        # [batch_size, hidden_size] -> [batch_size, vocab_size]
        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output('logits', self._dtype)

        if use_cache:
            for i, present in enumerate(presents):
                present.mark_output(f'present_key_value_{i}', self._kv_dtype)
            return (lm_logits, presents)

        return lm_logits

    def prepare_inputs(self, max_batch_size, max_input_len, max_new_tokens,
                       use_cache, max_beam_width):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        head_size = self._hidden_size // self._num_heads
        num_heads = self._num_heads // self._tensor_parallel
        max_len = max_input_len + max_new_tokens
        bb_range = [
            1, (max_batch_size * max_beam_width + 1) // 2,
            max_batch_size * max_beam_width
        ]
        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        beam_width_range = [1, (max_beam_width + 1) // 2, max_beam_width]
        inlen_range = [1, 1, max_input_len]
        max_len_range = [0, (max_len + 1) // 2, max_len]
        num_tokens_range = [
            1, max_batch_size * max_beam_width,
            max(max_input_len * max_batch_size, max_beam_width * max_batch_size)
        ]

        past_key_value = []
        sequence_length = None
        past_key_value_length = None
        masked_tokens = None
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        use_inflight_batching_gpt_attention_plugin = default_net(
        ).plugin_config.inflight_batching_gpt_attention_plugin
        remove_input_padding = default_net().plugin_config.remove_input_padding

        if remove_input_padding:
            input_ids = Tensor(name='input_ids',
                               dtype=trt.int32,
                               shape=[1, -1],
                               dim_range=OrderedDict([
                                   ('batch_size', [1]),
                                   ('num_tokens', [num_tokens_range]),
                               ]))
            position_ids = Tensor(name='position_ids',
                                  dtype=trt.int32,
                                  shape=[1, -1],
                                  dim_range=OrderedDict([
                                      ('batch_size', [1]),
                                      ('num_tokens', [num_tokens_range]),
                                  ]))
        else:
            input_ids = Tensor(name='input_ids',
                               dtype=trt.int32,
                               shape=[-1, -1],
                               dim_range=OrderedDict([
                                   ('batch_size', [bb_range]),
                                   ('input_len', [inlen_range]),
                               ]))
            position_ids = Tensor(name='position_ids',
                                  dtype=trt.int32,
                                  shape=[-1, -1],
                                  dim_range=OrderedDict([
                                      ('batch_size', [bb_range]),
                                      ('input_len', [inlen_range]),
                                  ]))

        past_key_value_pointers = []
        for i in range(self._num_layers):
            kv_dim_range = OrderedDict([
                ('batch_size', [bb_range]),
                ('kv', [2]),
                ('num_heads', [num_heads]),
                ('past_key_len', [max_len_range]),
                ('head_size', [head_size]),
            ])
            kv = Tensor(name=f'past_key_value_{i}',
                        dtype=self._kv_dtype,
                        shape=[-1, 2, num_heads, -1, head_size],
                        dim_range=kv_dim_range)
            past_key_value.append(kv)
            # TODO(kaiyu): Remove this when TRT fix the named dimension
            if not remove_input_padding:
                assertion(shape(input_ids, 0) == shape(kv, 0), 'batch size')

            if use_inflight_batching_gpt_attention_plugin:
                kv = Tensor(
                    name=f'past_key_value_pointers_{i}',
                    dtype=trt.int32,
                    # 2 INT32s for representing a single INT64 pointer
                    shape=[-1, 2],
                    dim_range=OrderedDict(batch_size=[bs_range],
                                          pointer_width=[2]))

                past_key_value_pointers.append(kv)

        if use_gpt_attention_plugin or use_inflight_batching_gpt_attention_plugin:
            past_key_value_length = Tensor(
                name='past_key_value_length',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('past_key_value_length',
                                        [max_len_range])]),
            )

        if use_gpt_attention_plugin:
            sequence_length = Tensor(
                name='sequence_length',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size', [bb_range])]),
            )
            masked_tokens = Tensor(name='masked_tokens',
                                   dtype=trt.int32,
                                   shape=[-1, -1],
                                   dim_range=OrderedDict([
                                       ('batch_size', [bb_range]),
                                       ('max_seq_len', [max_len_range]),
                                   ]))

        input_lengths = Tensor(name='input_lengths',
                               dtype=trt.int32,
                               shape=[-1],
                               dim_range=OrderedDict([('batch_size', [bb_range])
                                                      ]))

        max_input_length = Tensor(name='max_input_length',
                                  dtype=trt.int32,
                                  shape=[-1],
                                  dim_range=OrderedDict([('max_input_len',
                                                          [inlen_range])]))

        last_token_ids = Tensor(name='last_token_ids',
                                dtype=trt.int32,
                                shape=[-1],
                                dim_range=OrderedDict([
                                    ('batch_size', [bb_range]),
                                ]))
        input_ids_ragged = RaggedTensor.from_row_lengths(
            input_ids, input_lengths, max_input_length)

        cache_indirection = None
        if not use_inflight_batching_gpt_attention_plugin:
            cache_indirection = Tensor(name='cache_indirection',
                                       dtype=trt.int32,
                                       shape=[-1, -1, -1],
                                       dim_range=OrderedDict([
                                           ('batch_size', [bs_range]),
                                           ('beam_width', [beam_width_range]),
                                           ('max_seq_len', [max_len_range]),
                                       ]))

        inflight_batching_args = None
        if use_inflight_batching_gpt_attention_plugin:
            inflight_batching_args = InflightBatchingParam(
                # [nbReq]
                host_input_lengths=Tensor(
                    name='host_input_lengths',
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict(batch_size=[bs_range])),
                # [nbReq]
                host_beam_widths=Tensor(
                    name='beam_widths',
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict(batch_size=[bs_range])),
                # [nbReq, 2]
                cache_indir_pointers=Tensor(
                    name='cache_indir_pointers',
                    dtype=trt.int32,
                    # 2 INT32s for representing a single INT64 pointer
                    shape=[-1, 2],
                    dim_range=OrderedDict(batch_size=[bs_range],
                                          pointer_width=[2])),
                # [nbReq]
                host_req_cache_max_seq_lengths=Tensor(
                    name='req_cache_max_seq_lengths',
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict(batch_size=[bs_range])),
                max_input_length=max_input_len,
                max_beam_width=max_beam_width,
                use_int8_kv_cache=QuantMode(0),
                past_key_value_pointers=past_key_value_pointers)

        return (input_ids_ragged, position_ids, past_key_value, sequence_length,
                past_key_value_length, masked_tokens, True, last_token_ids,
                cache_indirection, inflight_batching_args)
