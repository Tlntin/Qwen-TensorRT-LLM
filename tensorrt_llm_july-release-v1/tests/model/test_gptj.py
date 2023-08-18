import os
import sys
import tempfile
import unittest
from itertools import product

import numpy as np
import tensorrt as trt
import torch
from parameterized import parameterized
from transformers import GPTJConfig, GPTJForCausalLM

import tensorrt_llm
from tensorrt_llm import Builder
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from examples.gptj.weight import load_from_hf_gpt_j


class TestGPTJ(unittest.TestCase):

    def _gen_hf_gpt_j(self, hidden_act, n_layer, max_length, dtype):
        gpt_config = GPTJConfig(activation_function=hidden_act,
                                n_layer=n_layer,
                                max_length=max_length,
                                torch_dtype=dtype,
                                n_embd=4096,
                                n_head=16,
                                rotary_dim=64)
        hf_gpt = GPTJForCausalLM(gpt_config).cuda().to(
            tensorrt_llm._utils.str_dtype_to_torch(dtype)).eval()
        return gpt_config, hf_gpt

    def _gen_tensorrt_llm_network(self, network, builder, hf_gpt, gpt_config,
                                  batch_size, beam_width, input_len, output_len,
                                  fp16, gpt_attention_plugin, tensor_parallel,
                                  apply_query_key_layer_scaling):
        num_layers = gpt_config.n_layer
        num_heads = gpt_config.n_head
        hidden_size = gpt_config.n_embd
        vocab_size = gpt_config.vocab_size
        hidden_act = gpt_config.activation_function
        n_positions = gpt_config.n_positions
        rotary_dim = gpt_config.rotary_dim
        tensor_parallel_group = list(range(tensor_parallel))

        with net_guard(network):
            kv_dtype = trt.float16 if fp16 else trt.float32
            # Initialize model
            tensorrt_llm_gpt = tensorrt_llm.models.GPTJForCausalLM(
                num_layers=num_layers,
                num_heads=num_heads,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                hidden_act=hidden_act,
                max_position_embeddings=n_positions,
                rotary_dim=rotary_dim,
                dtype=kv_dtype,
                tensor_parallel=tensor_parallel,  # TP only
                tensor_parallel_group=tensor_parallel_group,  # TP only
                apply_query_key_layer_scaling=apply_query_key_layer_scaling)
            inputs = tensorrt_llm_gpt.prepare_inputs(batch_size,
                                                     input_len,
                                                     output_len,
                                                     use_cache=True,
                                                     max_beam_width=beam_width)
            load_from_hf_gpt_j(tensorrt_llm_gpt, hf_gpt, fp16=fp16)

            # Prepare
            network.set_named_parameters(tensorrt_llm_gpt.named_parameters())

            tensorrt_llm_gpt(*inputs)

        return network

    def _gen_tensorrt_llm_runtime(self,
                                  log_level,
                                  dtype,
                                  world_size,
                                  rank,
                                  gpt_config,
                                  hf_gpt,
                                  model,
                                  use_attention_plugin,
                                  batch_size,
                                  beam_width,
                                  input_len,
                                  output_len,
                                  use_refit,
                                  use_ln_gemm_plugin,
                                  apply_query_key_layer_scaling,
                                  context_fmha_flag=ContextFMHAType.disabled,
                                  enable_remove_input_padding=False,
                                  use_in_flight_batching=False):
        tensorrt_llm.logger.set_level('error')
        mapping = tensorrt_llm.Mapping(world_size, rank)

        runtime = None
        builder = Builder()
        fp16 = (dtype == 'float16')

        with tempfile.TemporaryDirectory() as tmpdirname:
            network = builder.create_network()
            if use_attention_plugin:
                network.plugin_config.set_gpt_attention_plugin(dtype)
            if use_ln_gemm_plugin:
                network.plugin_config.set_gemm_plugin(dtype)
                network.plugin_config.set_layernorm_plugin(dtype)
            if enable_remove_input_padding:
                network.plugin_config.enable_remove_input_padding()
            if use_in_flight_batching:
                network.plugin_config.enable_in_flight_batching()
            network.plugin_config.set_context_fmha(context_fmha_flag)

            self._gen_tensorrt_llm_network(network, builder, hf_gpt, gpt_config,
                                           batch_size, beam_width, input_len,
                                           output_len, fp16,
                                           use_attention_plugin, world_size,
                                           apply_query_key_layer_scaling)

            builder_config = builder.create_builder_config(
                name='gptj',
                precision=dtype,
                timing_cache='model.cache',
                tensor_parallel=world_size,  # TP only
                use_refit=use_refit,
            )
            engine_buffer = builder.build_engine(network, builder_config)
            runtime = tensorrt_llm.runtime.generation._Runtime(
                engine_buffer, mapping)

            ok = builder.save_timing_cache(builder_config, 'model.cache')
            assert ok, "Failed to save timing cache."

        return runtime, engine_buffer

    def load_test_cases():
        test_cases = product([
            ContextFMHAType.disabled, ContextFMHAType.enabled,
            ContextFMHAType.enabled_with_fp32_acc
        ], [False, True])
        return test_cases

    @parameterized.expand(load_test_cases)
    def test_gptj_plugin(self, context_fmha_flag, enable_remove_input_padding):
        torch.random.manual_seed(0)
        use_refit = False
        apply_query_key_layer_scaling = False
        model = 'gptj'

        log_level = 'error'
        dtype = 'float16'
        world_size = 1
        rank = 0
        hidden_act = 'gelu'
        n_layer = 2
        max_length = 2
        batch_size = 1
        beam_width = 1
        seq_len = 12
        total_seq_len = max_length + seq_len
        use_attention_plugin = True
        use_ln_gemm_plugin = True
        use_in_flight_batching = False

        gpt_config, hf_gpt = self._gen_hf_gpt_j(hidden_act, n_layer,
                                                seq_len + max_length, dtype)
        runtime, _ = self._gen_tensorrt_llm_runtime(
            log_level,
            dtype,
            world_size,
            rank,
            gpt_config,
            hf_gpt,
            model,
            use_attention_plugin,
            batch_size,
            beam_width,
            seq_len,
            max_length,
            use_refit,
            use_ln_gemm_plugin,
            apply_query_key_layer_scaling,
            context_fmha_flag,
            enable_remove_input_padding=enable_remove_input_padding,
            use_in_flight_batching=use_in_flight_batching)
        key_value_cache_buffers = []
        head_size = gpt_config.n_embd // gpt_config.n_head
        for i in range(gpt_config.n_layer):
            key_value_cache_buffers.append(
                torch.zeros((
                    batch_size,
                    2,
                    gpt_config.n_head,
                    total_seq_len,
                    head_size,
                ),
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                            device='cuda'))
        sequence_length_buffer = torch.ones((batch_size, ),
                                            dtype=torch.int32,
                                            device='cuda')

        # compare context
        step = 0
        ctx_ids = torch.randint(100, (batch_size, seq_len)).int().cuda()

        with torch.no_grad():
            hf_outputs = hf_gpt.forward(ctx_ids, use_cache=True)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]

        ctx_input_lengths = seq_len * torch.ones(
            (batch_size), dtype=torch.int32, device='cuda')
        ctx_max_input_length = torch.zeros((seq_len, ),
                                           dtype=torch.int32,
                                           device='cuda')
        ctx_masked_tokens = torch.zeros((batch_size, seq_len),
                                        dtype=torch.int32,
                                        device='cuda')
        ctx_position_ids = torch.IntTensor(range(seq_len)).reshape(
            [1, seq_len]).expand([batch_size, seq_len]).cuda()
        ctx_last_token_ids = ctx_input_lengths.clone()

        if enable_remove_input_padding:
            ctx_ids = ctx_ids.view([1, batch_size * seq_len])
            ctx_position_ids = ctx_position_ids.view([1, batch_size * seq_len])
            ctx_last_token_ids = torch.cumsum(ctx_last_token_ids, dim=0).int()

        cache_indirections = [
            torch.full((
                batch_size,
                beam_width,
                total_seq_len,
            ),
                       0,
                       dtype=torch.int32,
                       device='cuda'),
            torch.full((
                batch_size,
                beam_width,
                total_seq_len,
            ),
                       0,
                       dtype=torch.int32,
                       device='cuda')
        ]  # ping-pong buffers

        ctx_shape = {
            'input_ids': ctx_ids.shape,
            'input_lengths': ctx_input_lengths.shape,
            'max_input_length': ctx_max_input_length.shape,
            'masked_tokens': ctx_masked_tokens.shape,
            'position_ids': ctx_position_ids.shape,
            'last_token_ids': ctx_last_token_ids.shape,
            'cache_indirection': cache_indirections[0].shape,
        }
        ctx_buffer = {
            'input_ids': ctx_ids,
            'input_lengths': ctx_input_lengths,
            'max_input_length': ctx_max_input_length,
            'masked_tokens': ctx_masked_tokens,
            'position_ids': ctx_position_ids,
            'last_token_ids': ctx_last_token_ids,
            'cache_indirection': cache_indirections[0],
        }
        shape = (batch_size, 2, gpt_config.n_head, total_seq_len,
                 gpt_config.n_embd // gpt_config.n_head)
        for i in range(gpt_config.n_layer):
            ctx_shape[f'past_key_value_{i}'] = shape
            ctx_buffer[f'past_key_value_{i}'] = key_value_cache_buffers[i]
            ctx_buffer[f'present_key_value_{i}'] = key_value_cache_buffers[i]
        ctx_buffer['sequence_length'] = sequence_length_buffer * (seq_len +
                                                                  step)
        ctx_shape['sequence_length'] = ctx_buffer['sequence_length'].shape
        ctx_buffer['past_key_value_length'] = torch.tensor([0, 1],
                                                           dtype=torch.int32)
        ctx_shape['past_key_value_length'] = ctx_buffer[
            'past_key_value_length'].shape

        context = runtime.context_0
        runtime._set_shape(context, ctx_shape)
        runtime._set_buffer(context, ctx_buffer)

        runtime._run(context)
        torch.cuda.synchronize()
        res = ctx_buffer['logits']

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   res.cpu().numpy(),
                                   atol=1e-1)

        v_inner = 16 // (2 if dtype == 'float16' else 4)
        for i in range(gpt_config.n_layer):
            res_present_key_value = ctx_buffer[f'present_key_value_{i}']

            past_key_value_tensor = res_present_key_value.permute(1, 0, 2, 3, 4)
            key, value = past_key_value_tensor.chunk(2)

            # TRT-LLM has the same cache layout for key and value:
            # [bs, n_head, max_seq_len, head_size]
            head_size = gpt_config.n_embd // gpt_config.n_head
            key = key.reshape(batch_size, gpt_config.n_head, total_seq_len,
                              head_size)

            value = value.reshape(batch_size, gpt_config.n_head, total_seq_len,
                                  head_size)

            ref_present_key, ref_present_value = hf_outputs.past_key_values[i]

            np.testing.assert_allclose(ref_present_key.cpu().numpy(),
                                       key[:, :, :seq_len, :].cpu().numpy(),
                                       atol=1e-1)
            np.testing.assert_allclose(ref_present_value.cpu().numpy(),
                                       value[:, :, :seq_len, :].cpu().numpy(),
                                       atol=1e-1)

        # compare generation
        step = 1
        step1_id = torch.randint(100, (batch_size, 1)).int().cuda()
        gen_position_ids = torch.ones_like(step1_id).int().cuda() * seq_len
        gen_input_lengths = ctx_input_lengths.clone()
        gen_max_input_length = ctx_max_input_length.clone()
        gen_masked_tokens = torch.zeros((batch_size, total_seq_len),
                                        dtype=torch.int32,
                                        device='cuda')
        gen_last_token_ids = torch.zeros_like(gen_input_lengths).int().cuda()

        with torch.no_grad():
            hf_outputs = hf_gpt.forward(
                step1_id,
                past_key_values=hf_outputs.past_key_values,
                position_ids=gen_position_ids,
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
            'max_input_length': gen_max_input_length.shape,
            'masked_tokens': gen_masked_tokens.shape,
            'position_ids': gen_position_ids.shape,
            'last_token_ids': gen_last_token_ids.shape,
            'cache_indirection': cache_indirections[1].shape,
        }
        step1_buffer = {
            'input_ids': step1_id,
            'input_lengths': gen_input_lengths,
            'max_input_length': gen_max_input_length,
            'masked_tokens': gen_masked_tokens,
            'position_ids': gen_position_ids,
            'last_token_ids': gen_last_token_ids,
            'cache_indirection': cache_indirections[1],
        }
        for i in range(gpt_config.n_layer):
            step1_shape[f'past_key_value_{i}'] = shape
        step1_shape['sequence_length'] = (batch_size, )
        step1_shape['past_key_value_length'] = (2, )
        for i in range(gpt_config.n_layer):
            step1_buffer[f'past_key_value_{i}'] = key_value_cache_buffers[i]
            step1_buffer[f'present_key_value_{i}'] = key_value_cache_buffers[i]
        step1_buffer['sequence_length'] = sequence_length_buffer * (seq_len +
                                                                    step)
        step1_buffer['past_key_value_length'] = torch.tensor(
            [seq_len + step - 1, 0], dtype=torch.int32)

        context = runtime.context_1
        runtime._set_shape(context, step1_shape)
        runtime._set_buffer(context, step1_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = step1_buffer['logits']

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   res.cpu().numpy(),
                                   atol=1e-1)

    def test_gptj_noplugin_unsupported(self):

        use_refit = False
        apply_query_key_layer_scaling = False
        model = 'gptj'

        log_level = 'error'
        dtype = 'float16'
        world_size = 1
        rank = 0
        hidden_act = 'gelu'
        n_layer = 1
        max_length = 2
        batch_size = 4
        seq_len = 128
        use_attention_plugin = False
        use_ln_gemm_plugin = True
        beam_width = 1

        gpt_config, hf_gpt = self._gen_hf_gpt_j(hidden_act, n_layer,
                                                seq_len + max_length, dtype)
        with self.assertRaisesRegex(
                ValueError,
                ".*GPT-J RoPE is only supported with GPTAttention and ibGPTAttention plugin.*"
        ):
            runtime, _ = self._gen_tensorrt_llm_runtime(
                log_level, dtype, world_size, rank, gpt_config, hf_gpt, model,
                use_attention_plugin, batch_size, beam_width, seq_len,
                max_length, use_refit, use_ln_gemm_plugin,
                apply_query_key_layer_scaling)

        use_ln_gemm_plugin = False
        if trt.__version__[:3] == '8.6':
            with self.assertRaisesRegex(
                    AssertionError,
                    "You need to enable the LayerNorm plugin for GPT-J with TensorRT"
            ):
                runtime, _ = self._gen_tensorrt_llm_runtime(
                    log_level, dtype, world_size, rank, gpt_config, hf_gpt,
                    model, use_attention_plugin, batch_size, beam_width,
                    seq_len, max_length, use_refit, use_ln_gemm_plugin,
                    apply_query_key_layer_scaling)


if __name__ == '__main__':
    unittest.main()
