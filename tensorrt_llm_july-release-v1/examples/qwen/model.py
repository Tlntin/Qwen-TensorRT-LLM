from collections import OrderedDict

import tensorrt as trt

from tensorrt_llm._common import default_net
from tensorrt_llm._utils import pad_vocab_size, str_dtype_to_trt
from tensorrt_llm.functional import (RaggedTensor, Tensor, assertion, expand_mask,
                           gather_last_token_logits, shape)
from tensorrt_llm.layers import (Attention, AttentionMaskType, ColumnLinear, Embedding,
                       GatedMLP, PositionEmbeddingType, RmsNorm)
from tensorrt_llm.module import Module, ModuleList


class QWenBlock(Module):

    def __init__(self,
                 layer_id,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 dtype=None,
                 hidden_act='silu',
                 mlp_hidden_size=None,
                 neox_rotary_style=True,
                 multi_query_mode=False,
                 tp_group=None,
                 tp_size=1):
        super().__init__()
        self._layer_id = layer_id  # useful for debugging
        self.ln_1 = RmsNorm(normalized_shape=hidden_size,
                                       dtype=dtype)

        self.attention = Attention(
            hidden_size,
            num_attention_heads,
            max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=False,
            position_embedding_type=PositionEmbeddingType.rope,
            neox_rotary_style=neox_rotary_style,
            multi_query_mode=multi_query_mode,
            tp_group=tp_group,
            tp_size=tp_size)
        if not mlp_hidden_size:
            mlp_hidden_size = hidden_size * 4
        self.mlp = GatedMLP(hidden_size=hidden_size,
                            ffn_hidden_size=mlp_hidden_size // 2,
                            hidden_act=hidden_act,
                            dtype=dtype,
                            bias=False,
                            tp_group=tp_group,
                            tp_size=tp_size)
        self.ln_2 = RmsNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(self,
                hidden_states: RaggedTensor,
                attention_mask=None,
                past_key_value=None,
                sequence_length=None,
                past_key_value_length=None,
                masked_tokens=None,
                use_cache=False,
                cache_indirection=None):
        residual = hidden_states.data
        input_lengths = hidden_states.row_lengths
        max_input_length = hidden_states.max_row_length
        hidden_states = self.ln_1(hidden_states.data)

        attention_output = self.attention(
            RaggedTensor.from_row_lengths(hidden_states, input_lengths,
                                          max_input_length),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            sequence_length=sequence_length,
            past_key_value_length=past_key_value_length,
            masked_tokens=masked_tokens,
            use_cache=use_cache,
            cache_indirection=cache_indirection)

        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output.data

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        hidden_states = RaggedTensor.from_row_lengths(
            hidden_states, attention_output.row_lengths,
            attention_output.max_row_length)
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class QWenModel(Module):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 dtype,
                 mlp_hidden_size=None,
                 neox_rotary_style=True,
                 tensor_parallel=1,
                 tensor_parallel_group=None,
                 multi_query_mode=False):
        super().__init__()
        self.vocab_embedding = Embedding(vocab_size, hidden_size, dtype=dtype)

        self.layers = ModuleList([
            QWenBlock(layer_id=i,
                              hidden_size=hidden_size,
                              num_attention_heads=num_heads,
                              max_position_embeddings=max_position_embeddings,
                              dtype=dtype,
                              hidden_act=hidden_act,
                              mlp_hidden_size=mlp_hidden_size,
                              neox_rotary_style=neox_rotary_style,
                              multi_query_mode=multi_query_mode,
                              tp_group=tensor_parallel_group,
                              tp_size=tensor_parallel)
            for i in range(num_layers)
        ])

        self.ln_f = RmsNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(self,
                input_ids: RaggedTensor,
                position_ids=None,
                past_key_value=None,
                sequence_length=None,
                past_key_value_length=None,
                masked_tokens=None,
                use_cache=False,
                attention_mask=None,
                cache_indirection=None):

        hidden_states = self.vocab_embedding(input_ids.data)

        if past_key_value is None:
            past_key_value = tuple([None] * len(self.layers))

        if use_cache:
            presents = []

        if attention_mask is not None:
            attention_mask = expand_mask(attention_mask,
                                         shape(input_ids.data, -1))

        hidden_states = RaggedTensor.from_row_lengths(hidden_states,
                                                      input_ids.row_lengths,
                                                      input_ids.max_row_length)

        for layer, past in zip(self.layers, past_key_value):
            hidden_states = layer(hidden_states,
                                  past_key_value=past,
                                  sequence_length=sequence_length,
                                  past_key_value_length=past_key_value_length,
                                  masked_tokens=masked_tokens,
                                  use_cache=use_cache,
                                  attention_mask=attention_mask,
                                  cache_indirection=cache_indirection)

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        hidden_states = self.ln_f(hidden_states.data)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class QWenForCausalLM(QWenModel):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 dtype,
                 mlp_hidden_size=None,
                 neox_rotary_style=True,
                 tensor_parallel=1,
                 tensor_parallel_group=None,
                 multi_query_mode=False):
        if isinstance(dtype, str):
            self.kv_dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self.kv_dtype = dtype
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tensor_parallel = tensor_parallel
        self._multi_query_mode = multi_query_mode
        super().__init__(num_layers, num_heads, hidden_size, vocab_size,
                         hidden_act, max_position_embeddings, dtype,
                         mlp_hidden_size, neox_rotary_style, tensor_parallel,
                         tensor_parallel_group, multi_query_mode)
        vocab_size_padded = pad_vocab_size(vocab_size, tensor_parallel)
        self.lm_head = ColumnLinear(hidden_size,
                                    vocab_size_padded,
                                    bias=False,
                                    dtype=dtype,
                                    tp_group=tensor_parallel_group,
                                    tp_size=tensor_parallel,
                                    gather_output=True)

    def forward(self,
                input_ids: RaggedTensor,
                position_ids=None,
                past_key_value=None,
                sequence_length=None,
                past_key_value_length=None,
                masked_tokens=None,
                use_cache=False,
                last_token_ids=None,
                attention_mask=None,
                cache_indirection=None):
        hidden_states = super().forward(input_ids, position_ids, past_key_value,
                                        sequence_length, past_key_value_length,
                                        masked_tokens, use_cache,
                                        attention_mask, cache_indirection)

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = gather_last_token_logits(
            hidden_states, last_token_ids,
            default_net().plugin_config.remove_input_padding)

        # [batch_size, hidden_size] -> [batch_size, vocab_size]
        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output('logits', self.kv_dtype)

        if use_cache:
            for i, present in enumerate(presents):
                present.mark_output(f'present_key_value_{i}', self.kv_dtype)
            return (lm_logits, presents)

        return lm_logits

    def prepare_inputs(self, max_batch_size, max_input_len, max_new_tokens,
                       use_cache, max_beam_width):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        head_size = self.hidden_size // self.num_heads
        num_heads = self.num_heads // self.tensor_parallel
        num_heads_kv = 1 if self._multi_query_mode else num_heads
        max_len = max_input_len + max_new_tokens
        bb_range = [
            1, (max_batch_size * max_beam_width + 1) // 2,
            max_batch_size * max_beam_width
        ]
        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        beam_width_range = [1, (max_beam_width + 1) // 2, max_beam_width]
        inlen_range = [1, 1, max_input_len]
        max_len_range = [0, (max_len + 1) // 2, max_len]
        mask_len_range = [1, (max_len + 1) // 2 + 1, max_len + 1]
        num_tokens_range = [
            1, max_batch_size * max_beam_width,
            max(max_input_len * max_batch_size, max_beam_width * max_batch_size)
        ]

        past_key_value = []
        sequence_length = None
        past_key_value_length = None
        masked_tokens = None
        attention_mask = None
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
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

        for i in range(self.num_layers):
            kv_dim_range = OrderedDict([
                ('batch_size', [bb_range]),
                ('kv', [2]),
                ('num_heads', [num_heads_kv]),
                ('past_key_len', [max_len_range]),
                ('head_size', [head_size]),
            ])
            kv = Tensor(name=f'past_key_value_{i}',
                        dtype=self.kv_dtype,
                        shape=[-1, 2, num_heads_kv, -1, head_size],
                        dim_range=kv_dim_range)
            past_key_value.append(kv)
            # TODO(kaiyu): Remove this when TRT fix the named dimension
            if not remove_input_padding:
                assertion(shape(input_ids, 0) == shape(kv, 0), 'batch size')

        if use_gpt_attention_plugin:
            sequence_length = Tensor(
                name='sequence_length',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size', [bb_range])]),
            )
            past_key_value_length = Tensor(
                name='past_key_value_length',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('past_key_value_length',
                                        [max_len_range])]),
            )
            masked_tokens = Tensor(name='masked_tokens',
                                   dtype=trt.int32,
                                   shape=[-1, -1],
                                   dim_range=OrderedDict([
                                       ('batch_size', [bb_range]),
                                       ('max_seq_len', [max_len_range]),
                                   ]))
        else:
            attention_mask = Tensor(name='attention_mask',
                                    dtype=trt.int32,
                                    shape=[-1, -1],
                                    dim_range=OrderedDict([
                                        ('batch_size', [bb_range]),
                                        ('mask_len', [mask_len_range]),
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

        cache_indirection = Tensor(name='cache_indirection',
                                   dtype=trt.int32,
                                   shape=[-1, -1, -1],
                                   dim_range=OrderedDict([
                                       ('batch_size', [bs_range]),
                                       ('beam_width', [beam_width_range]),
                                       ('max_seq_len', [max_len_range]),
                                   ]))

        return (input_ids_ragged, position_ids, past_key_value, sequence_length,
                past_key_value_length, masked_tokens, True, last_token_ids,
                attention_mask, cache_indirection)
