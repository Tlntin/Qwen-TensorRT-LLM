import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import tensorrt as trt
import torch
from cuda import cudart

from .._utils import pad_vocab_size, trt_dtype_to_torch
from ..logger import logger
from ..mapping import Mapping
from .kv_cache_manager import GenerationSequence, KVCacheManager
from .session import _scoped_stream
from .tensor import RaggedTensor


def _prepare_attention_mask(
    input_ids: torch.Tensor,
    pad_id: Optional[int] = None,
):
    if pad_id is not None:
        return input_ids.ne(pad_id).int()
    else:
        return torch.ones(input_ids.shape,
                          dtype=torch.int32,
                          device=input_ids.device)


def _tile_beam_width(tensor: torch.Tensor, num_beams: int):
    new_shape = np.array(tensor.shape)
    new_shape[0] = new_shape[0] * num_beams

    tile_size = np.ones(new_shape.shape, dtype=np.int32)
    tile_size = np.insert(tile_size, 1, num_beams)

    new_tensor = torch.unsqueeze(tensor, 1)
    new_tensor = new_tensor.tile(tile_size.tolist())
    new_tensor = new_tensor.reshape(new_shape.tolist())
    return new_tensor


class _Runtime(object):
    runtime_rank: int
    runtime: trt.Runtime
    engine: trt.ICudaEngine
    context_0: trt.IExecutionContext
    context_1: trt.IExecutionContext

    def __init__(self, engine_buffer, mapping: Mapping):
        self.__prepare(mapping, engine_buffer)

    def __create_and_setup_context(self, address, profile_idx,
                                   stream) -> trt.IExecutionContext:
        context = self.engine.create_execution_context_without_device_memory()
        assert context is not None
        context.device_memory = address
        context.set_optimization_profile_async(profile_idx, stream)
        return context

    def __prepare(self, mapping: Mapping, engine_buffer):
        self.runtime_rank = mapping.rank
        local_rank = self.runtime_rank % mapping.gpus_per_node
        torch.cuda.set_device(local_rank)
        status, = cudart.cudaSetDevice(local_rank)
        assert status == 0, status

        self.runtime = trt.Runtime(logger.trt_logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_buffer)
        assert self.engine is not None
        status, address = cudart.cudaMalloc(self.engine.device_memory_size)
        assert status == 0, status
        with _scoped_stream() as stream:
            self.context_0 = self.__create_and_setup_context(address, 0, stream)
            self.context_1 = self.__create_and_setup_context(address, 0, stream)

    def _set_shape(self, context: trt.IExecutionContext,
                   shape_dict: Dict[str, List[int]]):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                context.set_input_shape(name, shape_dict[name])

    def _set_buffer(self, context: trt.IExecutionContext,
                    buffer_dict: Dict[str, torch.Tensor]):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if name not in buffer_dict.keys():
                dtype = self.engine.get_tensor_dtype(name)
                shape = context.get_tensor_shape(name)
                buffer_dict[name] = torch.zeros(tuple(shape),
                                                dtype=trt_dtype_to_torch(dtype),
                                                device='cuda')
            context.set_tensor_address(name, buffer_dict[name].data_ptr())

    def _run(self, context: trt.IExecutionContext, stream=None) -> bool:
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        ok = context.execute_async_v3(stream)
        return ok


@dataclass
class ModelConfig:
    vocab_size: int
    num_layers: int
    num_heads: int
    hidden_size: int
    gpt_attention_plugin: bool
    inflight_batching_gpt_attention_plugin: bool = False
    multi_query_mode: bool = False
    remove_input_padding: bool = False
    model_name: str = ""
    paged_kv_cache: bool = False
    tokens_per_block: int = 64
    use_prompt_tuning: bool = False


@dataclass
class SamplingConfig:
    end_id: int
    pad_id: int

    num_beams: int = field(default=1)
    temperature: float = field(default=1.0)
    top_k: int = field(default=1)
    top_p: float = field(default=0.0)
    length_penalty: float = field(default=1)
    repetition_penalty: float = field(default=1)
    min_length: int = field(default=1)
    presence_penalty: float = field(default=0.0)

    ## None here means user didn't set it, and dynamicDecodeOp.cpp take optional value
    ## The real default value is set in dynamicDecodeOp.cpp when it's None
    beam_search_diversity_rate: float = field(init=False, default=None)
    random_seed: int = field(init=False, default=None)
    output_cum_log_probs: bool = field(init=False, default=False)
    output_log_probs: bool = field(init=False, default=False)


class GenerationSession(object):

    _model_config: ModelConfig
    mapping: Mapping
    runtime: _Runtime
    device: torch.device
    batch_size: int
    buffer_allocated: bool
    debug_mode: bool

    def __init__(self,
                 model_config: ModelConfig,
                 engine_buffer,
                 mapping: Mapping,
                 debug_mode=False):
        assert isinstance(model_config, ModelConfig)
        self._model_config = model_config
        runtime = _Runtime(engine_buffer, mapping)

        self.mapping = mapping
        self.runtime = runtime
        self.device = torch.device(
            f'cuda:{runtime.runtime_rank % mapping.gpus_per_node}')
        torch.cuda.set_device(self.device)
        self.debug_mode = debug_mode

        # Optional inputs for dynamic decoder
        self.top_p_decay = None
        self.top_p_min = None
        self.top_p_reset_ids = None
        #TODO: in tensorrt_llm/cpp/tensorrt_llm/thop/dynamicDecodeOp.cpp it's T, can be float or half?
        self.embedding_bias_opt = None
        self.stop_words_list = None
        self.bad_words_list = None

        self.buffer = None
        self.buffer_allocated = False

        pp_size = 1
        self.vocab_size_padded = pad_vocab_size(self.vocab_size,
                                                self.mapping.tp_size)
        self.dynamic_decoder = torch.classes.FasterTransformer.DynamicDecodeOp(
            self.vocab_size, self.vocab_size_padded, self.mapping.tp_size,
            pp_size, torch.float32)

        self.gather_tree = torch.ops.tensorrt_llm.gather_tree

        expected_tensor_names = ['input_ids'] \
            + [f'past_key_value_{i}' for i in range(model_config.num_layers)] \
            + ['logits'] \
            + [f'present_key_value_{i}' for i in range(model_config.num_layers)] \
            + ['input_lengths'] \
            + ['position_ids'] \
            + ['last_token_ids'] \
            + ['max_input_length']

        if not model_config.inflight_batching_gpt_attention_plugin:
            expected_tensor_names += ['cache_indirection']

        if self.paged_kv_cache:
            expected_tensor_names += [
                f'kv_cache_block_pointers_{i}' for i in range(self.num_layers)
            ]

        if model_config.gpt_attention_plugin:
            expected_tensor_names += [
                'sequence_length', 'past_key_value_length', 'masked_tokens'
            ]
        elif model_config.inflight_batching_gpt_attention_plugin:
            expected_tensor_names += [
                f'past_key_value_pointers_{i}' for i in range(self.num_layers)
            ]
            expected_tensor_names += [
                'host_input_lengths', 'past_key_value_length', 'beam_widths',
                'req_cache_max_seq_lengths'
            ]
        else:
            expected_tensor_names += ['attention_mask']

        if model_config.use_prompt_tuning:
            expected_tensor_names += [
                'prompt_embedding_table', 'tasks', 'prompt_vocab_size'
            ]

        found_tensor_names = [
            runtime.engine.get_tensor_name(i)
            for i in range(runtime.engine.num_io_tensors)
        ]
        if not self.debug_mode and set(expected_tensor_names) != set(
                found_tensor_names):
            logger.error(
                f"The following expected tensors are not found: {set(expected_tensor_names).difference(set(found_tensor_names))}"
            )
            logger.error(
                f"Those tensors in engine are not expected: {set(found_tensor_names).difference(set(expected_tensor_names))}"
            )
            logger.error(f"Expected tensor names: {expected_tensor_names}")
            logger.error(f"Found tensor names: {found_tensor_names}")
            raise RuntimeError(
                "Tensor names in engine are not the same as expected, to use this GenerationSession, " \
                    "you need to use GPTLMHeadModel.prepare_inputs to create TRT Network inputs."
            )

    @property
    def vocab_size(self):
        return self._model_config.vocab_size

    @property
    def num_layers(self):
        return self._model_config.num_layers

    @property
    def num_heads(self):
        return self._model_config.num_heads

    @property
    def hidden_size(self):
        return self._model_config.hidden_size

    @property
    def use_gpt_attention_plugin(self):
        return self._model_config.gpt_attention_plugin

    @property
    def use_inflight_batching_gpt_attention_plugin(self):
        return self._model_config.use_inflight_batching_gpt_attention_plugin

    @property
    def multi_query_mode(self):
        return self._model_config.multi_query_mode

    @property
    def paged_kv_cache(self):
        return self._model_config.paged_kv_cache

    @property
    def tokens_per_block(self):
        return self._model_config.tokens_per_block

    @property
    def remove_input_padding(self):
        return self._model_config.remove_input_padding

    @property
    def num_heads_kv(self):
        return 1 if self.multi_query_mode else self.num_heads

    @property
    def head_size(self):
        return self.hidden_size // self.num_heads

    def __setup_decoder(self, input_ids: torch.Tensor,
                        sampling_config: SamplingConfig,
                        input_lengths: torch.Tensor):
        '''Allocate buffers and setup the post-processing decoder kernel
        '''
        batch_size = input_lengths.shape[0]
        scfg = sampling_config  # just to make a shorter name, no other meaning
        self.top_k = torch.full([batch_size], scfg.top_k, dtype=torch.int32)
        self.top_p = torch.full([batch_size], scfg.top_p, dtype=torch.float32)
        self.temperature = torch.full([batch_size],
                                      scfg.temperature,
                                      dtype=torch.float32)
        self.repetition_penalty = torch.full([batch_size],
                                             scfg.repetition_penalty,
                                             dtype=torch.float32)
        if scfg.repetition_penalty == 1.0:
            self.repetition_penalty = None

        self.length_penalty = torch.FloatTensor([scfg.length_penalty])

        self.presence_penalty = torch.full([batch_size],
                                           scfg.presence_penalty,
                                           dtype=torch.float32)
        if scfg.presence_penalty == 0.0:
            self.presence_penalty = None
        assert (
            scfg.presence_penalty == 0.0 or scfg.repetition_penalty == 0.0
        ), f"presence_penalty({scfg.presence_penalty}) and repetition_penalty({scfg.repetition_penalty}) cannot be larger than 0.0 at the same time."
        self.min_length = torch.full([batch_size],
                                     scfg.min_length,
                                     dtype=torch.int32)

        if scfg.beam_search_diversity_rate is not None:
            self.beam_search_diversity_rate = torch.full(
                [batch_size],
                scfg.beam_search_diversity_rate,
                dtype=torch.float32)
        else:
            self.beam_search_diversity_rate = None

        if scfg.random_seed is not None:
            self.random_seed = torch.full([batch_size],
                                          scfg.random_seed,
                                          dtype=torch.int64)
        else:
            self.random_seed = None

        self.dynamic_decoder.setup(
            batch_size, scfg.num_beams, self.top_k, self.top_p,
            self.temperature, self.repetition_penalty, self.presence_penalty,
            self.min_length, self.length_penalty,
            self.beam_search_diversity_rate, self.random_seed, self.top_p_decay,
            self.top_p_min, self.top_p_reset_ids)

        assert scfg.end_id is not None, "end_id cannot be none"
        assert scfg.pad_id is not None, 'pad_id cannot be none'
        self.end_ids = torch.full((batch_size * scfg.num_beams, ),
                                  scfg.end_id,
                                  dtype=torch.int32,
                                  device=self.device)
        max_input_length = input_lengths.max()

        if input_ids.shape[0] != input_lengths.shape[0]:
            # dim 0 of input_ids is not batch size, which means remove_padding is enabled
            split_ids_list = list(
                torch.split(input_ids,
                            input_lengths.cpu().numpy().tolist(),
                            dim=1))
            padded_input_ids = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(split_ids_list, dtype=torch.int32),
                scfg.pad_id).cuda().reshape(batch_size, max_input_length)
        else:
            padded_input_ids = input_ids
        if scfg.num_beams > 1:
            tiled_input_ids = _tile_beam_width(padded_input_ids, scfg.num_beams)
            tiled_input_ids = tiled_input_ids.reshape(batch_size,
                                                      scfg.num_beams,
                                                      max_input_length)
            transposed_input_ids = tiled_input_ids.permute(2, 0, 1)
            self.output_ids = torch.cat(
                (transposed_input_ids,
                 torch.zeros(self.max_seq_length - max_input_length,
                             batch_size,
                             scfg.num_beams,
                             dtype=padded_input_ids.dtype,
                             device=padded_input_ids.device)))
        else:
            transposed_input_ids = padded_input_ids.permute(1, 0)
            self.output_ids = torch.cat(
                (transposed_input_ids,
                 torch.zeros(self.max_seq_length - max_input_length,
                             batch_size,
                             dtype=padded_input_ids.dtype,
                             device=padded_input_ids.device)))

        self.parent_ids = torch.zeros(
            (self.max_seq_length, batch_size, scfg.num_beams),
            dtype=torch.int32,
            device=self.device)

        if scfg.num_beams > 1 or scfg.output_cum_log_probs:
            self.cum_log_probs = torch.full((batch_size, scfg.num_beams),
                                            -1e20,
                                            dtype=torch.float32,
                                            device=self.device)
            self.cum_log_probs[:, 0] = 0.0
        else:
            self.cum_log_probs = None

        if scfg.output_log_probs:
            self.log_probs = torch.zeros(
                (self.max_new_tokens, batch_size, scfg.num_beams),
                dtype=torch.float32,
                device=self.device)
        else:
            self.log_probs = None

        self.finished = torch.zeros((batch_size, scfg.num_beams),
                                    dtype=torch.bool,
                                    device=self.device)

    def setup(self, batch_size: int, max_input_length: int,
              max_new_tokens: int):
        # Store these params related to buffer size to check against
        # the input shape with the params given in decode()
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_input_length + max_new_tokens

        def tensor_dtype(name):
            # return torch dtype given tensor name for convenience
            dtype = trt_dtype_to_torch(
                self.runtime.engine.get_tensor_dtype(name))
            return dtype

        self.buffer = {
            'logits':
            torch.empty((batch_size, self.vocab_size_padded),
                        dtype=tensor_dtype('logits'),
                        device=self.device),
            'max_input_length':
            torch.empty((max_input_length, ),
                        dtype=tensor_dtype('max_input_length'),
                        device=self.device)
        }

        if self.paged_kv_cache:
            blocks = math.ceil(batch_size * self.max_seq_length /
                               self.tokens_per_block)
            cache_shape = (
                blocks,
                2,
                self.num_heads_kv,
                self.tokens_per_block,
                self.head_size,
            )
        else:
            cache_shape = (
                batch_size,
                2,
                self.num_heads_kv,
                self.max_seq_length,
                self.head_size,
            )
        for i in range(self.num_layers):
            self.buffer[f'present_key_value_{i}'] = torch.empty(
                cache_shape,
                dtype=tensor_dtype(f'present_key_value_{i}'),
                device=self.device)
        if self.use_gpt_attention_plugin:
            self.sequence_length_buffer = torch.ones((batch_size, ),
                                                     dtype=torch.int32,
                                                     device=self.device)
        else:
            # We need two set of kv cache buffers,
            # one for inputs, and the other for outputs.
            # They will take turns to act as input and output buffers.
            for i in range(self.num_layers):
                self.buffer[f'1_present_key_value_{i}'] = torch.empty(
                    cache_shape,
                    dtype=tensor_dtype(f'present_key_value_{i}'),
                    device=self.device)

        # Init KV cache block manager
        if self.paged_kv_cache:
            max_blocks_per_seq = math.ceil(self.max_seq_length /
                                           self.tokens_per_block)
            memory_pools = [
                self.buffer[f'present_key_value_{i}']
                for i in range(self.num_layers)
            ]
            self.kv_cache_manager = KVCacheManager(memory_pools, blocks,
                                                   self.tokens_per_block,
                                                   max_blocks_per_seq)

        self.buffer_allocated = True

    def _get_context_shape_buffer(self,
                                  input_ids: torch.Tensor,
                                  max_input_length: int,
                                  step: int,
                                  masked_tokens: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  last_token_ids: torch.Tensor,
                                  attention_mask: torch.Tensor,
                                  cache_indirection: torch.Tensor,
                                  kv_cache_block_pointers: List[torch.Tensor],
                                  prompt_embedding_table: torch.Tensor = None,
                                  tasks: torch.Tensor = None,
                                  prompt_vocab_size: torch.Tensor = None):
        ctx_shape = {
            'input_ids': input_ids.shape,
            'input_lengths': input_lengths.shape,
            'position_ids': position_ids.shape,
            'last_token_ids': last_token_ids.shape,
            'max_input_length': self.buffer['max_input_length'].shape,
            'cache_indirection': cache_indirection.shape,
        }
        ctx_buffer = {
            'input_ids': input_ids.contiguous(),
            'logits': self.buffer['logits'],
            'input_lengths': input_lengths.contiguous(),
            'position_ids': position_ids.contiguous(),
            'last_token_ids': last_token_ids.contiguous(),
            'max_input_length': self.buffer['max_input_length'],
            'cache_indirection': cache_indirection.contiguous(),
        }
        if prompt_embedding_table is not None:
            ctx_buffer[
                'prompt_embedding_table'] = prompt_embedding_table.contiguous()
            ctx_shape['prompt_embedding_table'] = prompt_embedding_table.shape

            ctx_buffer['tasks'] = tasks.contiguous()
            ctx_shape['tasks'] = tasks.shape

            ctx_buffer['prompt_vocab_size'] = prompt_vocab_size.contiguous()
            ctx_shape['prompt_vocab_size'] = prompt_vocab_size.shape

        if self.paged_kv_cache:
            for idx in range(self.num_layers):
                ctx_buffer[
                    f'kv_cache_block_pointers_{idx}'] = kv_cache_block_pointers[
                        idx].contiguous()
                ctx_shape[
                    f'kv_cache_block_pointers_{idx}'] = kv_cache_block_pointers[
                        idx].shape

        batch_size = input_lengths.shape[0]
        for idx in range(self.num_layers):
            if not self.use_gpt_attention_plugin:
                kv_cache_shape = (batch_size, 2, self.num_heads_kv, 0,
                                  self.head_size)
                # for empty tensor, TRT does not really use the tensor data, so any dtype is fine
                kv_cache_buffer = torch.zeros((1, ),
                                              dtype=torch.float32,
                                              device=self.device)
                ctx_shape.update({
                    f'past_key_value_{idx}': kv_cache_shape,
                })
                ctx_buffer.update({
                    f'past_key_value_{idx}':
                    kv_cache_buffer,
                    f'present_key_value_{idx}':
                    self.buffer[f'present_key_value_{idx}'],
                })
            else:
                cache_shape = self.buffer[f'present_key_value_{idx}'].shape
                key_value_cache = self.buffer[f'present_key_value_{idx}']
                ctx_shape.update({
                    f'past_key_value_{idx}': cache_shape,
                })
                ctx_buffer.update({
                    f'past_key_value_{idx}': key_value_cache,
                    f'present_key_value_{idx}': key_value_cache
                })
        if self.use_gpt_attention_plugin:
            ctx_shape.update({
                'sequence_length': (batch_size, ),
                'past_key_value_length': (2, ),
                'masked_tokens': masked_tokens.shape,
            })
            ctx_buffer.update({
                'sequence_length':
                self.sequence_length_buffer * (max_input_length + step - 0),
                'past_key_value_length':
                torch.tensor([0, 1], dtype=torch.int32),
                'masked_tokens':
                masked_tokens.contiguous(),
            })
        else:
            ctx_shape.update({'attention_mask': attention_mask.shape})
            ctx_buffer.update({'attention_mask': attention_mask.contiguous()})
        return ctx_shape, ctx_buffer

    def _get_next_step_shape_buffer(self,
                                    batch_size: int,
                                    beam_width: int,
                                    max_input_length: int,
                                    step: int,
                                    masked_tokens: torch.Tensor,
                                    input_lengths: torch.Tensor,
                                    position_ids: torch.Tensor,
                                    last_token_ids: torch.Tensor,
                                    attention_mask: torch.Tensor,
                                    cache_indirection: torch.Tensor,
                                    kv_cache_block_pointers: List[torch.Tensor],
                                    prompt_embedding_table: torch.Tensor = None,
                                    tasks: torch.Tensor = None,
                                    prompt_vocab_size: torch.Tensor = None):
        next_step_shape = {
            'input_ids':
            (1, batch_size * beam_width) if self.remove_input_padding else
            (batch_size * beam_width, 1),
            'input_lengths':
            input_lengths.shape,
            'position_ids':
            position_ids.shape,
            'last_token_ids':
            last_token_ids.shape,
            'max_input_length':
            self.buffer['max_input_length'].shape,
            'cache_indirection':
            cache_indirection.shape,
        }
        next_step_buffer = {
            'input_ids': self.output_ids[step + max_input_length],
            'logits': self.buffer['logits'],
            'input_lengths': input_lengths.contiguous(),
            'position_ids': position_ids.contiguous(),
            'last_token_ids': last_token_ids.contiguous(),
            'max_input_length': self.buffer['max_input_length'],
            'cache_indirection': cache_indirection.contiguous(),
        }

        if self.paged_kv_cache:
            for idx in range(self.num_layers):
                next_step_buffer[
                    f'kv_cache_block_pointers_{idx}'] = kv_cache_block_pointers[
                        idx].contiguous()
                next_step_shape[
                    f'kv_cache_block_pointers_{idx}'] = kv_cache_block_pointers[
                        idx].shape

        if prompt_embedding_table is not None:
            next_step_buffer[
                'prompt_embedding_table'] = prompt_embedding_table.contiguous()
            next_step_shape[
                'prompt_embedding_table'] = prompt_embedding_table.shape

            next_step_buffer['tasks'] = tasks.contiguous()
            next_step_shape['tasks'] = tasks.shape

            next_step_buffer[
                'prompt_vocab_size'] = prompt_vocab_size.contiguous()
            next_step_shape['prompt_vocab_size'] = prompt_vocab_size.shape

        for idx in range(self.num_layers):
            if not self.use_gpt_attention_plugin:
                if step % 2:
                    next_step_buffer.update({
                        f'past_key_value_{idx}':
                        self.buffer[f'1_present_key_value_{idx}'],
                        f'present_key_value_{idx}':
                        self.buffer[f'present_key_value_{idx}'],
                    })
                else:
                    next_step_buffer.update({
                        f'past_key_value_{idx}':
                        self.buffer[f'present_key_value_{idx}'],
                        f'present_key_value_{idx}':
                        self.buffer[f'1_present_key_value_{idx}'],
                    })
                next_shape = (batch_size * beam_width, 2, self.num_heads_kv,
                              max_input_length + step, self.head_size)
                next_step_shape[f'past_key_value_{idx}'] = next_shape
            else:
                cache_shape = self.buffer[f'present_key_value_{idx}'].shape
                key_value_cache = self.buffer[f'present_key_value_{idx}']
                next_step_buffer.update({
                    f'past_key_value_{idx}':
                    key_value_cache,
                    f'present_key_value_{idx}':
                    key_value_cache,
                })
                next_step_shape[f'past_key_value_{idx}'] = cache_shape
        if self.use_gpt_attention_plugin:
            next_step_shape.update({
                'sequence_length': (batch_size * beam_width, ),
                'past_key_value_length': (2, ),
                'masked_tokens': masked_tokens.shape,
            })
            next_step_buffer.update({
                'sequence_length':
                self.sequence_length_buffer * (max_input_length + step),
                'past_key_value_length':
                torch.tensor([max_input_length + step, 0], dtype=torch.int32),
                'masked_tokens':
                masked_tokens.contiguous(),
            })
        else:
            next_step_shape.update({'attention_mask': attention_mask.shape})
            next_step_buffer.update({
                'attention_mask':
                attention_mask.contiguous(),
            })
        return next_step_shape, next_step_buffer

    def _prepare_context_inputs(self, batch_size, input_lengths,
                                use_gpt_attention_plugin, remove_input_padding,
                                **kwargs):

        last_token_ids = input_lengths.detach().clone()
        if use_gpt_attention_plugin:
            max_input_length = kwargs.pop('max_input_length')
            if remove_input_padding:
                position_ids = torch.unsqueeze(
                    torch.concat([
                        torch.cuda.IntTensor(range(input_lengths[i]))
                        for i in range(batch_size)
                    ]), 0)
                last_token_ids = torch.cumsum(last_token_ids, dim=0).int()
            else:
                position_ids = torch.cuda.IntTensor(
                    range(max_input_length)).reshape([1, -1]).expand(
                        [batch_size, -1])
            return {
                'position_ids': position_ids,
                'last_token_ids': last_token_ids
            }
        else:
            input_ids = kwargs.pop('input_ids')
            pad_id = kwargs.pop('pad_id', None)
            attention_mask = _prepare_attention_mask(input_ids, pad_id)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.int()

            return {
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'last_token_ids': last_token_ids
            }

    def _prepare_generation_inputs(self, batch_size, input_lengths,
                                   use_gpt_attention_plugin,
                                   remove_input_padding, **kwargs):
        last_token_ids = torch.ones_like(input_lengths)
        if use_gpt_attention_plugin:
            step = kwargs.pop('step')
            position_ids = input_lengths + step
            if remove_input_padding:
                position_ids = torch.unsqueeze(position_ids, 0)
                last_token_ids = torch.cumsum(last_token_ids, dim=0).int()
            else:
                position_ids = torch.unsqueeze(position_ids, 1)

            return {
                'position_ids': position_ids,
                'last_token_ids': last_token_ids
            }
        else:
            attention_mask = kwargs.pop('attention_mask')
            num_beams = kwargs.pop('num_beams')
            attention_mask = torch.cat((attention_mask,
                                        attention_mask.new_ones(
                                            (batch_size * num_beams, 1))),
                                       dim=-1).contiguous()
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids[:, -1].unsqueeze(-1)
            position_ids = position_ids.int()

            return {
                'position_ids': position_ids,
                'last_token_ids': last_token_ids,
                'attention_mask': attention_mask,
            }

    def decode_ragged(self, input_ids: RaggedTensor,
                      sampling_config: SamplingConfig):
        return self.decode(input_ids.data, input_ids.row_lengths,
                           sampling_config)

    def decode_batch(self, input_ids: Sequence[torch.Tensor],
                     sampling_config: SamplingConfig):
        input_ids = RaggedTensor.from_tensors(input_ids)
        return self.decode_ragged(input_ids, sampling_config)

    def decode(self,
               input_ids: torch.Tensor,
               input_lengths: torch.Tensor,
               sampling_config: SamplingConfig,
               prompt_embedding_table: torch.Tensor = None,
               tasks: torch.Tensor = None,
               prompt_vocab_size: torch.Tensor = None):
        batch_size = input_lengths.size(0)
        max_input_length = torch.max(input_lengths).item()
        assert batch_size == self.batch_size, \
            "Given batch size is different from the one used in setup()," \
            "rerun the setup function with the new batch size to avoid buffer overflow."
        assert max_input_length == self.max_input_length, \
            "Given input length is large then the one used in setup()," \
            "rerun the setup function with the new max_input_length to avoid buffer overflow."
        ite = 0  # index of local batches, will always be 0 if pp_size = 1
        scfg = sampling_config

        self.__setup_decoder(input_ids, scfg, input_lengths)
        if not self.buffer_allocated:
            raise RuntimeError('Buffer not allocated, please call setup first!')

        sequence_limit_lengths = torch.full((batch_size, 1),
                                            self.max_seq_length,
                                            dtype=torch.int32,
                                            device=self.device)
        sequence_lengths = torch.full((batch_size * scfg.num_beams, 1),
                                      max_input_length,
                                      dtype=torch.int32,
                                      device=self.device)
        len_list = torch.arange(0,
                                self.max_seq_length,
                                dtype=torch.int32,
                                device=self.device).unsqueeze(0).expand(
                                    batch_size, -1)
        mask = (len_list >= input_lengths.unsqueeze(1)) & (len_list <
                                                           max_input_length)
        masked_tokens = torch.zeros((batch_size, self.max_seq_length),
                                    dtype=torch.int32,
                                    device=self.device).masked_fill_(mask, 1)

        cache_indirections = [
            torch.full((
                batch_size,
                scfg.num_beams,
                self.max_seq_length,
            ),
                       0,
                       dtype=torch.int32,
                       device=self.device),
            torch.full((
                batch_size,
                scfg.num_beams,
                self.max_seq_length,
            ),
                       0,
                       dtype=torch.int32,
                       device=self.device)
        ]  # ping-pong buffers

        if self.paged_kv_cache:
            # Add sequences to the manager
            for bi in range(batch_size):
                generation_sequence = GenerationSequence(seq_idx=bi,
                                                         batch_idx=bi)
                self.kv_cache_manager.add_sequence(generation_sequence,
                                                   input_ids.size(1))

        kv_cache_block_pointers = []
        # start context phase
        for step in range(0, self.max_new_tokens):
            if self.paged_kv_cache:
                kv_cache_block_pointers = self.kv_cache_manager.get_pointer_arrays(
                )

            if step % 2:
                context = self.runtime.context_0
                this_src_cache_indirection = cache_indirections[1]
                this_tgt_cache_indirection = cache_indirections[0]
                next_src_cache_indirection = cache_indirections[0]
            else:
                context = self.runtime.context_1
                this_src_cache_indirection = cache_indirections[0]
                this_tgt_cache_indirection = cache_indirections[1]
                next_src_cache_indirection = cache_indirections[1]

            if step == 0:
                model_inputs = self._prepare_context_inputs(
                    batch_size=batch_size,
                    input_lengths=input_lengths,
                    use_gpt_attention_plugin=self.use_gpt_attention_plugin,
                    remove_input_padding=self.remove_input_padding,
                    max_input_length=max_input_length,
                    input_ids=input_ids,
                    pad_id=scfg.pad_id)

                position_ids = model_inputs.get('position_ids')
                last_token_ids = model_inputs.get('last_token_ids')
                attention_mask = model_inputs.get('attention_mask', None)

                ctx_shape, ctx_buffer = self._get_context_shape_buffer(
                    input_ids, max_input_length, step, masked_tokens,
                    input_lengths, position_ids, last_token_ids, attention_mask,
                    this_src_cache_indirection, kv_cache_block_pointers,
                    prompt_embedding_table, tasks, prompt_vocab_size)
                self.runtime._set_shape(context, ctx_shape)
                self.runtime._set_buffer(context, ctx_buffer)

            # dynamic_decoder currently use torch's current stream, so must let TRT enqueue use same stream here
            stream = torch.cuda.current_stream().cuda_stream
            ok = self.runtime._run(context, stream)
            if not ok:
                raise RuntimeError('Executing TRT engine failed!')
            if self.debug_mode:
                torch.cuda.synchronize()

            if step == 0 and scfg.num_beams > 1:

                if not self.use_gpt_attention_plugin:
                    attention_mask = _tile_beam_width(attention_mask,
                                                      scfg.num_beams)
                input_lengths = _tile_beam_width(input_lengths, scfg.num_beams)
                if self.use_gpt_attention_plugin:
                    self.sequence_length_buffer = _tile_beam_width(
                        self.sequence_length_buffer, scfg.num_beams)
                masked_tokens = _tile_beam_width(masked_tokens, scfg.num_beams)

                # Move tiling before logit computing of context
                for key in self.buffer.keys():
                    if "present_key_value" in key:
                        self.buffer[key] = _tile_beam_width(
                            self.buffer[key], scfg.num_beams)
                self.buffer['logits'] = _tile_beam_width(
                    self.buffer['logits'], scfg.num_beams)

            if not step == self.max_new_tokens - 1:
                # Set shape and address for the next step
                model_inputs = self._prepare_generation_inputs(
                    batch_size=batch_size,
                    input_lengths=input_lengths,
                    use_gpt_attention_plugin=self.use_gpt_attention_plugin,
                    remove_input_padding=self.remove_input_padding,
                    step=step,
                    num_beams=scfg.num_beams,
                    attention_mask=attention_mask,
                )

                position_ids = model_inputs.get('position_ids')
                last_token_ids = model_inputs.get('last_token_ids')
                attention_mask = model_inputs.get('attention_mask', None)

                next_context = self.runtime.context_1 if step % 2 else self.runtime.context_0
                next_step_shape, next_step_buffer = self._get_next_step_shape_buffer(
                    batch_size, scfg.num_beams, max_input_length, step,
                    masked_tokens, input_lengths, position_ids, last_token_ids,
                    attention_mask, next_src_cache_indirection,
                    kv_cache_block_pointers, prompt_embedding_table, tasks,
                    prompt_vocab_size)
                self.runtime._set_shape(next_context, next_step_shape)
                self.runtime._set_buffer(next_context, next_step_buffer)

            logits = self.buffer['logits']
            if logits is not None:
                # [batch_size x scft.num_beams, vocab_size_padded] -> [batch_size, scfg.num_beams, vocab_size_padded]
                next_token_logits = logits.reshape(
                    (batch_size, scfg.num_beams, -1)).to(torch.float32)
                decode_step = step + max_input_length
                should_stop = self.dynamic_decoder.forward(
                    next_token_logits, decode_step, max_input_length, ite,
                    batch_size, self.end_ids, self.top_k, self.top_p,
                    self.temperature, self.repetition_penalty,
                    self.presence_penalty, self.min_length, self.length_penalty,
                    self.beam_search_diversity_rate, self.top_p_decay,
                    self.top_p_min, self.top_p_reset_ids,
                    self.embedding_bias_opt, input_lengths,
                    sequence_limit_lengths, self.stop_words_list,
                    self.bad_words_list, this_src_cache_indirection,
                    self.output_ids, self.finished, sequence_lengths,
                    self.cum_log_probs, self.log_probs, self.parent_ids,
                    this_tgt_cache_indirection)

                if should_stop.item():
                    if self.paged_kv_cache:
                        # Free all blocks in all sequences.
                        # With in-flight batching and while loop we'll free some sequences, when they are done
                        self.kv_cache_manager.step([True] * batch_size *
                                                   scfg.num_beams)

                    # output shape of self.gather_tree: [batch_size, beam_width, output_len]
                    final_output_ids = self.gather_tree(
                        sequence_lengths, self.output_ids, self.parent_ids,
                        self.end_ids, input_lengths, batch_size, scfg.num_beams,
                        max_input_length, self.max_seq_length)
                    return final_output_ids

            if self.paged_kv_cache and step < self.max_new_tokens - 1:
                # Iterate to the next step in KV cache manager.
                # Increase number of tokens for all unfinished sequences.
                # And allocate new blocks if needed.
                # We set this to False for all sequences, since we use only length criterion to stop now
                self.kv_cache_manager.step([False] * batch_size *
                                           scfg.num_beams)

        if self.paged_kv_cache:
            # Free all blocks in all sequences.
            # With in-flight batching and while loop we'll free some sequences, when they are done
            self.kv_cache_manager.step([True] * batch_size * scfg.num_beams)

        # output shape of self.gather_tree: [batch_size, beam_width, output_len]
        final_output_ids = self.gather_tree(sequence_lengths, self.output_ids,
                                            self.parent_ids, self.end_ids,
                                            input_lengths, batch_size,
                                            scfg.num_beams, max_input_length,
                                            self.max_seq_length)

        return final_output_ids


class ChatGLM6BHeadModelGenerationSession(GenerationSession):

    def _prepare_context_inputs(self, batch_size, input_lengths,
                                use_gpt_attention_plugin, remove_input_padding,
                                **kwargs):

        assert use_gpt_attention_plugin
        assert not remove_input_padding
        last_token_ids = input_lengths.detach().clone()
        max_input_length = kwargs.pop('max_input_length')
        position_ids = torch.zeros([batch_size, 2, max_input_length],
                                   dtype=torch.int32)
        position_ids[:, 0, :] = torch.arange(max_input_length)
        for i in range(batch_size):
            position_ids[i, 0, max_input_length - 1] = max_input_length - 2
            position_ids[i, 1, max_input_length - 1] = 1
            position_ids[i, :, input_lengths[i]:] = 0
        position_ids = position_ids.cuda()
        return {'position_ids': position_ids, 'last_token_ids': last_token_ids}

    def _prepare_generation_inputs(self, batch_size, input_lengths,
                                   use_gpt_attention_plugin,
                                   remove_input_padding, **kwargs):
        assert use_gpt_attention_plugin
        assert not remove_input_padding
        last_token_ids = torch.ones_like(input_lengths)

        step = kwargs.pop('step')
        num_beams = kwargs.pop('num_beams')

        data = []
        for i in range(batch_size):
            data.append([[input_lengths[i * num_beams] - 2], [step + 2]])
        position_ids = torch.cuda.IntTensor(data).int()

        return {'position_ids': position_ids, 'last_token_ids': last_token_ids}
