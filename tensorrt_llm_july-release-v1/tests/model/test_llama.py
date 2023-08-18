import os
import sys
import tempfile
import unittest
from itertools import product

import numpy as np
import torch
from parameterized import parameterized
from transformers import LlamaConfig, LlamaForCausalLM

import tensorrt_llm
from tensorrt_llm import Builder
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from examples.llama.weight import load_from_hf_llama


class TestLLaMA(unittest.TestCase):
    EOS_TOKEN = 2
    PAD_TOKEN = 2

    def _gen_tensorrt_llm_network(self, network, hf_llama,
                                  llama_config: LlamaConfig, batch_size,
                                  beam_width, input_len, output_len, dtype,
                                  multi_query_mode, rank, tensor_parallel):
        tensor_parallel_group = list(range(tensor_parallel))

        with net_guard(network):
            kv_dtype = str_dtype_to_trt(dtype)

            # Initialize model
            tensorrt_llm_llama = tensorrt_llm.models.LLaMAForCausalLM(
                num_layers=llama_config.num_hidden_layers,
                num_heads=llama_config.num_attention_heads,
                hidden_size=llama_config.hidden_size,
                vocab_size=llama_config.vocab_size,
                hidden_act=llama_config.hidden_act,
                max_position_embeddings=llama_config.max_position_embeddings,
                dtype=kv_dtype,
                mlp_hidden_size=llama_config.intermediate_size,
                neox_rotary_style=True,
                tensor_parallel=tensor_parallel,  # TP only
                tensor_parallel_group=tensor_parallel_group,  # TP only
                multi_query_mode=multi_query_mode,
            )
            load_from_hf_llama(tensorrt_llm_llama,
                               hf_llama,
                               dtype=dtype,
                               rank=rank,
                               tensor_parallel=tensor_parallel,
                               multi_query_mode=multi_query_mode)
            # Prepare
            network.set_named_parameters(tensorrt_llm_llama.named_parameters())
            inputs = tensorrt_llm_llama.prepare_inputs(batch_size, input_len,
                                                       output_len, True,
                                                       beam_width)
            # Forward
            tensorrt_llm_llama(*inputs)

        return network

    def _gen_tensorrt_llm_engine(self,
                                 dtype,
                                 rank,
                                 world_size,
                                 llama_config,
                                 hf_llama,
                                 model_name,
                                 use_plugin,
                                 batch_size,
                                 beam_width,
                                 input_len,
                                 output_len,
                                 use_refit,
                                 fast_building=False,
                                 context_fmha_flag=ContextFMHAType.disabled,
                                 enable_remove_input_padding=False,
                                 multi_query_mode=False):

        builder = Builder()

        with tempfile.TemporaryDirectory() as tmpdirname:
            network = builder.create_network()
            if use_plugin:
                network.plugin_config.set_gpt_attention_plugin(dtype)
            if fast_building:
                network.plugin_config.set_gemm_plugin(dtype)
            if enable_remove_input_padding:
                network.plugin_config.enable_remove_input_padding()
            network.plugin_config.set_context_fmha(context_fmha_flag)

            self._gen_tensorrt_llm_network(network, hf_llama, llama_config,
                                           batch_size, beam_width, input_len,
                                           output_len, dtype, multi_query_mode,
                                           rank, world_size)

            builder_config = builder.create_builder_config(
                name=model_name,
                precision=dtype,
                timing_cache='model.cache',
                tensor_parallel=world_size,  # TP only
                use_refit=use_refit,
                multi_query_mode=multi_query_mode,
            )
            engine_buffer = builder.build_engine(network, builder_config)
            return engine_buffer

    def _gen_tensorrt_llm_runtime(self,
                                  log_level,
                                  dtype,
                                  world_size,
                                  rank,
                                  llama_config,
                                  hf_llama,
                                  model_name,
                                  use_plugin,
                                  batch_size,
                                  beam_width,
                                  input_len,
                                  output_len,
                                  use_refit,
                                  fast_building=False,
                                  context_fmha_flag=ContextFMHAType.disabled,
                                  enable_remove_input_padding=False,
                                  multi_query_mode=False):
        tensorrt_llm.logger.set_level(log_level)
        mapping = tensorrt_llm.Mapping(world_size, rank)
        engine_buffer = self._gen_tensorrt_llm_engine(
            dtype, rank, world_size, llama_config, hf_llama, model_name,
            use_plugin, batch_size, beam_width, input_len, output_len,
            use_refit, fast_building, context_fmha_flag,
            enable_remove_input_padding, multi_query_mode)
        runtime = tensorrt_llm.runtime.generation._Runtime(
            engine_buffer, mapping)
        return runtime, engine_buffer

    def load_test_cases():
        test_cases = list(
            product([False], [False, True], [
                ContextFMHAType.disabled, ContextFMHAType.enabled,
                ContextFMHAType.enabled_with_fp32_acc
            ], [False, True], ['float16'], [False]))
        test_cases.append(
            (False, True, ContextFMHAType.disabled, False, 'bfloat16', False))
        test_cases.append((False, True, ContextFMHAType.enabled, False,
                           'float16', True))  # needs transformers==4.31.0
        return test_cases

    @parameterized.expand(load_test_cases)
    def test_llama(self, use_refit, fast_building, context_fmha_flag,
                   enable_remove_input_padding, dtype, multi_query_mode):
        model = 'llama'
        log_level = 'error'
        use_plugin = True  # gpt plugin
        batch_size = 4
        beam_width = 1
        input_len = 4
        output_len = 2
        max_seq_len = input_len + output_len
        world_size = 1
        rank = 0
        llama_config = LlamaConfig()
        llama_config.hidden_act = 'silu'
        llama_config.num_hidden_layers = 2
        llama_config.max_position_embeddings = 64
        llama_config.vocab_size = 128
        llama_config.hidden_size = 64
        llama_config.intermediate_size = 24
        llama_config.num_attention_heads = 2
        if hasattr(llama_config, "num_key_value_heads"):
            llama_config.num_key_value_heads = world_size if multi_query_mode else llama_config.num_attention_heads
            print(llama_config.num_key_value_heads)
        llama_config.pad_token_id = self.PAD_TOKEN
        llama_config.eos_token_id = self.EOS_TOKEN
        hf_llama = LlamaForCausalLM(llama_config).cuda()
        runtime, _ = self._gen_tensorrt_llm_runtime(
            log_level, dtype, world_size, rank, llama_config, hf_llama, model,
            use_plugin, batch_size, beam_width, input_len, output_len,
            use_refit, fast_building, context_fmha_flag,
            enable_remove_input_padding, multi_query_mode)
        key_value_cache_buffers = []
        head_size = llama_config.hidden_size // llama_config.num_attention_heads
        for i in range(llama_config.num_hidden_layers):
            key_value_cache_buffers.append(
                torch.zeros((
                    batch_size,
                    2,
                    llama_config.num_attention_heads,
                    max_seq_len,
                    head_size,
                ),
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                            device='cuda'))
        sequence_length_buffer = torch.ones((batch_size, ),
                                            dtype=torch.int32,
                                            device='cuda')

        # compare context
        step = 0
        ctx_ids = torch.randint(100, (batch_size, input_len)).int().cuda()
        ctx_input_lengths = input_len * torch.ones(
            (batch_size), dtype=torch.int32, device='cuda')
        ctx_masked_tokens = torch.zeros((batch_size, input_len),
                                        dtype=torch.int32,
                                        device='cuda')
        ctx_position_ids = torch.IntTensor(range(input_len)).reshape(
            [1, input_len]).expand([batch_size, input_len]).cuda()
        ctx_last_token_ids = ctx_input_lengths.clone()
        ctx_max_input_length = torch.zeros((input_len, ),
                                           dtype=torch.int32,
                                           device='cuda')

        with torch.no_grad():
            hf_outputs = hf_llama.forward(ctx_ids)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]

        if enable_remove_input_padding:
            ctx_ids = ctx_ids.view([1, batch_size * input_len])
            ctx_position_ids = ctx_position_ids.view(
                [1, batch_size * input_len])
            ctx_last_token_ids = torch.cumsum(ctx_last_token_ids, dim=0).int()

        cache_indirections = [
            torch.full((
                batch_size,
                beam_width,
                max_seq_len,
            ),
                       0,
                       dtype=torch.int32,
                       device='cuda'),
            torch.full((
                batch_size,
                beam_width,
                max_seq_len,
            ),
                       0,
                       dtype=torch.int32,
                       device='cuda')
        ]  # ping-pong buffers

        ctx_shape = {
            'input_ids': ctx_ids.shape,
            'input_lengths': ctx_input_lengths.shape,
            'masked_tokens': ctx_masked_tokens.shape,
            'position_ids': ctx_position_ids.shape,
            'last_token_ids': ctx_last_token_ids.shape,
            'max_input_length': ctx_max_input_length.shape,
            'cache_indirection': cache_indirections[0].shape,
        }
        ctx_buffer = {
            'input_ids': ctx_ids,
            'input_lengths': ctx_input_lengths,
            'masked_tokens': ctx_masked_tokens,
            'position_ids': ctx_position_ids,
            'last_token_ids': ctx_last_token_ids,
            'max_input_length': ctx_max_input_length,
            'cache_indirection': cache_indirections[0],
        }
        kv_shape = (batch_size, 2, llama_config.num_key_value_heads,
                    max_seq_len, llama_config.hidden_size //
                    llama_config.num_attention_heads)
        for i in range(llama_config.num_hidden_layers):
            ctx_shape[f'past_key_value_{i}'] = kv_shape
            ctx_buffer[f'past_key_value_{i}'] = key_value_cache_buffers[i]
            ctx_buffer[f'present_key_value_{i}'] = key_value_cache_buffers[i]
        ctx_buffer['sequence_length'] = sequence_length_buffer * (input_len +
                                                                  step)
        ctx_shape['sequence_length'] = ctx_buffer['sequence_length'].shape
        ctx_shape['past_key_value_length'] = (2, )
        ctx_buffer['past_key_value_length'] = torch.tensor([0, 1],
                                                           dtype=torch.int32)

        context = runtime.context_0
        runtime._set_shape(context, ctx_shape)
        runtime._set_buffer(context, ctx_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = ctx_buffer['logits']

        np.testing.assert_allclose(ref.to(torch.float32).cpu().numpy(),
                                   res.to(torch.float32).cpu().numpy(),
                                   atol=1e-1)

        # compare generation
        step = 1
        step1_id = torch.randint(100, (batch_size, 1)).int().cuda()
        gen_input_lengths = ctx_input_lengths.clone()
        gen_max_input_length = ctx_max_input_length.clone()
        gen_masked_tokens = torch.zeros((batch_size, max_seq_len),
                                        dtype=torch.int32,
                                        device='cuda')
        gen_position_ids = torch.ones_like(step1_id).int().cuda() * input_len
        gen_last_token_ids = torch.zeros_like(gen_input_lengths).int().cuda()

        with torch.no_grad():
            hf_outputs = hf_llama.forward(
                step1_id,
                past_key_values=hf_outputs.past_key_values,
                use_cache=True)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]

        if enable_remove_input_padding:
            step1_id = step1_id.view([1, batch_size])
            gen_position_ids = gen_position_ids.view([1, batch_size])
            gen_last_token_ids = torch.ones_like(gen_input_lengths).int().cuda()
            gen_last_token_ids = torch.cumsum(gen_last_token_ids, dim=0).int()

        step1_shape = {
            'input_ids': step1_id.shape,
            'input_lengths': gen_input_lengths.shape,
            'masked_tokens': gen_masked_tokens.shape,
            'position_ids': gen_position_ids.shape,
            'last_token_ids': gen_last_token_ids.shape,
            'max_input_length': gen_max_input_length.shape,
            'cache_indirection': cache_indirections[1].shape,
        }
        step1_buffer = {
            'input_ids': step1_id,
            'input_lengths': gen_input_lengths,
            'masked_tokens': gen_masked_tokens,
            'position_ids': gen_position_ids,
            'last_token_ids': gen_last_token_ids,
            'max_input_length': gen_max_input_length.contiguous(),
            'cache_indirection': cache_indirections[1],
        }
        for i in range(llama_config.num_hidden_layers):
            step1_shape[f'past_key_value_{i}'] = kv_shape
        step1_shape['sequence_length'] = (batch_size, )
        step1_shape['past_key_value_length'] = (2, )
        for i in range(llama_config.num_hidden_layers):
            step1_buffer[f'past_key_value_{i}'] = key_value_cache_buffers[i]
            step1_buffer[f'present_key_value_{i}'] = key_value_cache_buffers[i]
        step1_buffer['sequence_length'] = sequence_length_buffer * (input_len +
                                                                    step)
        step1_buffer['past_key_value_length'] = torch.tensor(
            [input_len + step - 1, 0], dtype=torch.int32)

        context = runtime.context_1
        runtime._set_shape(context, step1_shape)
        runtime._set_buffer(context, step1_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = step1_buffer['logits']

        np.testing.assert_allclose(ref.to(torch.float32).cpu().numpy(),
                                   res.to(torch.float32).cpu().numpy(),
                                   atol=1e-1)


if __name__ == '__main__':
    unittest.main()
