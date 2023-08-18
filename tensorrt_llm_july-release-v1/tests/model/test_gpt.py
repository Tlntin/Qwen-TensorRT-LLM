import os
import sys
import tempfile
import unittest
from itertools import product

import numpy as np
import tensorrt as trt
import torch
from parameterized import parameterized
from transformers import GPT2Config, GPT2LMHeadModel

import tensorrt_llm
from tensorrt_llm import Builder
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.runtime import ModelConfig, RaggedTensor, SamplingConfig
from tensorrt_llm.runtime.generation import _prepare_attention_mask

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from examples.gpt.weight import load_from_hf_gpt


class TestGPT(unittest.TestCase):

    def _gen_hf_gpt(self, hidden_act, n_layer, max_length, dtype):
        gpt_config = GPT2Config(
            activation_function=hidden_act,
            n_layer=n_layer,
            max_length=max_length,
            torch_dtype=dtype,
        )
        hf_gpt = GPT2LMHeadModel(gpt_config).cuda().eval()
        return gpt_config, hf_gpt

    def _gen_tensorrt_llm_network(self, network, builder, hf_gpt, gpt_config,
                                  batch_size, input_len, output_len, fp16,
                                  gpt_attention_plugin, tensor_parallel,
                                  apply_query_key_layer_scaling):
        num_layers = gpt_config.n_layer
        num_heads = gpt_config.n_head
        hidden_size = gpt_config.n_embd
        vocab_size = gpt_config.vocab_size
        hidden_act = gpt_config.activation_function
        n_positions = gpt_config.n_positions
        tensor_parallel_group = list(range(tensor_parallel))

        with net_guard(network):
            kv_dtype = trt.float16 if fp16 else trt.float32
            # Initialize model
            tensorrt_llm_gpt = tensorrt_llm.models.GPTLMHeadModel(
                num_layers=num_layers,
                num_heads=num_heads,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                hidden_act=hidden_act,
                max_position_embeddings=n_positions,
                dtype=kv_dtype,
                tensor_parallel=tensor_parallel,  # TP only
                tensor_parallel_group=tensor_parallel_group,  # TP only
                apply_query_key_layer_scaling=apply_query_key_layer_scaling)
            inputs = tensorrt_llm_gpt.prepare_inputs(batch_size,
                                                     input_len,
                                                     output_len,
                                                     use_cache=True,
                                                     max_beam_width=1)
            load_from_hf_gpt(tensorrt_llm_gpt,
                             hf_gpt,
                             dtype="float16" if fp16 else "float32")

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
                                  use_plugin,
                                  batch_size,
                                  input_len,
                                  output_len,
                                  use_refit,
                                  fast_building=False,
                                  apply_query_key_layer_scaling=False,
                                  context_fmha_type=ContextFMHAType.disabled,
                                  enable_remove_input_padding=False):
        mapping = tensorrt_llm.Mapping(world_size, rank)

        runtime = None
        builder = Builder()
        fp16 = (dtype == 'float16')

        with tempfile.TemporaryDirectory() as tmpdirname:
            network = builder.create_network()
            if use_plugin:
                network.plugin_config.set_gpt_attention_plugin(dtype)
            if fast_building:
                network.plugin_config.set_gemm_plugin(dtype)
                network.plugin_config.set_layernorm_plugin(dtype)
            network.plugin_config.set_context_fmha(context_fmha_type)
            if enable_remove_input_padding:
                network.plugin_config.enable_remove_input_padding()

            self._gen_tensorrt_llm_network(network, builder, hf_gpt, gpt_config,
                                           batch_size, input_len, output_len,
                                           fp16, use_plugin, world_size,
                                           apply_query_key_layer_scaling)

            builder_config = builder.create_builder_config(
                name='gpt',
                precision=dtype,
                timing_cache='model.cache',
                tensor_parallel=world_size,  # TP only
                use_refit=use_refit,
            )
            engine_buffer = builder.build_engine(network, builder_config)
            runtime = tensorrt_llm.runtime.generation._Runtime(
                engine_buffer, mapping)
        return runtime, engine_buffer

    @parameterized.expand([(False)])
    def test_gpt_float32(self, use_refit):
        model = 'gpt'
        log_level = 'error'
        dtype = 'float32'
        world_size = 1
        rank = 0
        hidden_act = 'gelu'
        n_layer = 2
        max_length = 2
        batch_size = 4
        beam_width = 1
        seq_len = 128
        total_length = seq_len + max_length
        use_plugin = False

        gpt_config, hf_gpt = self._gen_hf_gpt(hidden_act, n_layer, max_length,
                                              dtype)
        runtime, _ = self._gen_tensorrt_llm_runtime(
            log_level, dtype, world_size, rank, gpt_config, hf_gpt, model,
            use_plugin, batch_size, seq_len, max_length, use_refit)

        # compare context
        pad_token_id = 50256
        ctx_ids = torch.randint(100, (batch_size, seq_len)).int().cuda()
        ctx_ids[0][-1] = pad_token_id
        ctx_ids[1][-3:] = pad_token_id
        ctx_ids[2][-5:] = pad_token_id
        ctx_input_lengths = seq_len * torch.ones(
            (batch_size), dtype=torch.int32, device='cuda')
        ctx_position_ids = torch.IntTensor(range(seq_len)).reshape(
            [1, seq_len]).expand([batch_size, seq_len]).cuda()
        ctx_last_token_ids = ctx_input_lengths.clone()
        ctx_attention_mask = _prepare_attention_mask(ctx_ids)
        ctx_max_input_length = torch.zeros((seq_len, ),
                                           dtype=torch.int32,
                                           device='cuda')

        cache_indirections = [
            torch.full((
                batch_size,
                beam_width,
                total_length,
            ),
                       0,
                       dtype=torch.int32,
                       device='cuda'),
            torch.full((
                batch_size,
                beam_width,
                total_length,
            ),
                       0,
                       dtype=torch.int32,
                       device='cuda')
        ]  # ping-pong buffers

        ctx_shape = {
            'input_ids': ctx_ids.shape,
            'position_ids': ctx_position_ids.shape,
            'input_lengths': ctx_input_lengths.shape,
            'last_token_ids': ctx_last_token_ids.shape,
            'attention_mask': ctx_attention_mask.shape,
            'max_input_length': ctx_max_input_length.shape,
            'cache_indirection': cache_indirections[0].shape,
        }
        ctx_buffer = {
            'input_ids': ctx_ids,
            'position_ids': ctx_position_ids,
            'input_lengths': ctx_input_lengths,
            'last_token_ids': ctx_last_token_ids,
            'attention_mask': ctx_attention_mask,
            'max_input_length': ctx_max_input_length,
            'cache_indirection': cache_indirections[0],
        }
        for i in range(gpt_config.n_layer):
            shape = (batch_size, 2, gpt_config.n_head, 0,
                     gpt_config.n_embd // gpt_config.n_head)
            past_buffer = torch.zeros((1, ),
                                      dtype=str_dtype_to_torch(dtype),
                                      device='cuda')
            ctx_shape.update({
                f'past_key_value_{i}': shape,
            })
            shape = (batch_size, 2, gpt_config.n_head, seq_len,
                     gpt_config.n_embd // gpt_config.n_head)
            ctx_buffer.update({
                f'past_key_value_{i}':
                past_buffer,
                f'present_key_value_{i}':
                torch.zeros(shape,
                            dtype=str_dtype_to_torch(dtype),
                            device='cuda'),
            })

        context = runtime.context_0
        runtime._set_shape(context, ctx_shape)
        runtime._set_buffer(context, ctx_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = ctx_buffer['logits']

        with torch.no_grad():
            hf_outputs = hf_gpt.forward(ctx_ids,
                                        attention_mask=ctx_attention_mask)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   res.cpu().numpy(),
                                   atol=1e-2)

        for i in range(gpt_config.n_layer):
            res_present_key_value = ctx_buffer[f'present_key_value_{i}']
            ref_present_key, ref_present_value = hf_outputs.past_key_values[i]

            past_key_value_tensor = res_present_key_value.permute(1, 0, 2, 3, 4)
            key, value = past_key_value_tensor.chunk(2)

            head_size = gpt_config.n_embd // gpt_config.n_head
            key = key.to(torch.float32).reshape(batch_size, gpt_config.n_head,
                                                seq_len, head_size)
            value = value.reshape(batch_size, gpt_config.n_head, seq_len,
                                  head_size)

            np.testing.assert_allclose(ref_present_key.cpu().numpy(),
                                       key.cpu().numpy(),
                                       atol=1e-2)

            np.testing.assert_allclose(ref_present_value.cpu().numpy(),
                                       value.cpu().numpy(),
                                       atol=1e-2)

        # compare generation
        gen_id = torch.randint(100, (batch_size, 1)).int().cuda()
        gen_input_lengths = ctx_input_lengths.clone()
        gen_max_input_length = ctx_max_input_length.clone()
        gen_masked_tokens = torch.zeros((batch_size, seq_len),
                                        dtype=torch.int32,
                                        device='cuda')
        gen_position_ids = torch.ones_like(gen_id).cuda() * seq_len
        gen_last_token_ids = torch.zeros_like(gen_input_lengths).cuda()
        gen_attention_mask = torch.cat([
            ctx_attention_mask,
            ctx_attention_mask.new_ones((ctx_attention_mask.shape[0], 1))
        ],
                                       dim=-1)
        step1_shape = {
            'input_ids': gen_id.shape,
            'input_lengths': gen_input_lengths.shape,
            'masked_tokens': gen_masked_tokens.shape,
            'position_ids': gen_position_ids.shape,
            'last_token_ids': gen_last_token_ids.shape,
            'attention_mask': gen_attention_mask.shape,
            'max_input_length': gen_max_input_length.shape,
            'cache_indirection': cache_indirections[1].shape,
        }
        step1_buffer = {
            'input_ids': gen_id,
            'input_lengths': gen_input_lengths.contiguous(),
            'masked_tokens': gen_masked_tokens.contiguous(),
            'position_ids': gen_position_ids.contiguous(),
            'last_token_ids': gen_last_token_ids.contiguous(),
            'attention_mask': gen_attention_mask.contiguous(),
            'max_input_length': gen_max_input_length.contiguous(),
            'cache_indirection': cache_indirections[1].contiguous(),
        }
        for i in range(gpt_config.n_layer):
            shape = (batch_size, 2, gpt_config.n_head, seq_len,
                     gpt_config.n_embd // gpt_config.n_head)
            step1_shape.update({
                f'past_key_value_{i}': shape,
            })
            step1_buffer.update({
                f'past_key_value_{i}':
                ctx_buffer[f'present_key_value_{i}'],
            })

        context = runtime.context_1
        runtime._set_shape(context, step1_shape)
        runtime._set_buffer(context, step1_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = step1_buffer['logits']

        with torch.no_grad():
            hf_outputs = hf_gpt.forward(
                gen_id,
                attention_mask=gen_attention_mask,
                past_key_values=hf_outputs.past_key_values,
                use_cache=True)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   res.cpu().numpy(),
                                   atol=1e-2)

        for i in range(gpt_config.n_layer):
            res_present_key_value = step1_buffer[f'present_key_value_{i}']

            ref_present_key, ref_present_value = hf_outputs.past_key_values[i]

            past_key_value_tensor = res_present_key_value.permute(1, 0, 2, 3, 4)
            key, value = past_key_value_tensor.chunk(2)

            head_size = gpt_config.n_embd // gpt_config.n_head
            key = key.reshape(batch_size, gpt_config.n_head, seq_len + 1,
                              head_size)
            value = value.reshape(batch_size, gpt_config.n_head, seq_len + 1,
                                  head_size)

            np.testing.assert_allclose(ref_present_key.cpu().numpy(),
                                       key.cpu().numpy(),
                                       atol=1e-2)

            np.testing.assert_allclose(ref_present_value.cpu().numpy(),
                                       value.cpu().numpy(),
                                       atol=1e-2)

    def load_test_cases():
        test_cases = list(
            product([False, True], [False, True], [False, True], [
                ContextFMHAType.disabled, ContextFMHAType.enabled,
                ContextFMHAType.enabled_with_fp32_acc
            ], [False, True]))
        return test_cases

    @parameterized.expand(load_test_cases)
    def test_gpt_plugin(self, use_refit, fast_building,
                        apply_query_key_layer_scaling, context_fmha_type,
                        enable_remove_input_padding):
        model = 'gpt'
        log_level = 'error'
        dtype = 'float16'
        world_size = 1
        rank = 0
        hidden_act = 'gelu'
        n_layer = 1
        max_length = 2
        batch_size = 4
        beam_width = 1
        seq_len = 128
        total_length = seq_len + max_length
        use_plugin = True
        gpt_config, hf_gpt = self._gen_hf_gpt(hidden_act, n_layer,
                                              seq_len + max_length, dtype)
        runtime, _ = self._gen_tensorrt_llm_runtime(
            log_level, dtype, world_size, rank, gpt_config, hf_gpt, model,
            use_plugin, batch_size, seq_len, max_length, use_refit,
            fast_building, apply_query_key_layer_scaling, context_fmha_type,
            enable_remove_input_padding)
        key_value_cache_buffers = []
        value_cache_buffers = []
        head_size = gpt_config.n_embd // gpt_config.n_head
        for i in range(gpt_config.n_layer):
            key_value_cache_buffers.append(
                torch.zeros((
                    batch_size,
                    2,
                    gpt_config.n_head,
                    total_length,
                    head_size,
                ),
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                            device='cuda'))
            value_cache_buffers.append(
                torch.zeros((
                    batch_size,
                    gpt_config.n_head,
                    total_length,
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
        ctx_input_lengths = seq_len * torch.ones(
            (batch_size), dtype=torch.int32, device='cuda')
        ctx_masked_tokens = torch.zeros((batch_size, seq_len),
                                        dtype=torch.int32,
                                        device='cuda')
        ctx_position_ids = torch.IntTensor(range(seq_len)).reshape(
            [1, seq_len]).expand([batch_size, seq_len]).cuda()
        ctx_last_token_ids = ctx_input_lengths.clone()
        ctx_max_input_length = torch.zeros((seq_len, ),
                                           dtype=torch.int32,
                                           device='cuda')

        with torch.no_grad():
            hf_outputs = hf_gpt.forward(ctx_ids)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]

        if enable_remove_input_padding:
            ctx_ids = ctx_ids.view([1, batch_size * seq_len])
            ctx_position_ids = ctx_position_ids.view([1, batch_size * seq_len])
            ctx_last_token_ids = torch.cumsum(ctx_last_token_ids, dim=0).int()

        cache_indirections = [
            torch.full((
                batch_size,
                beam_width,
                total_length,
            ),
                       0,
                       dtype=torch.int32,
                       device='cuda'),
            torch.full((
                batch_size,
                beam_width,
                total_length,
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
        shape = (batch_size, 2, gpt_config.n_head, total_length,
                 gpt_config.n_embd // gpt_config.n_head)
        for i in range(gpt_config.n_layer):
            ctx_shape[f'past_key_value_{i}'] = shape
            ctx_buffer[f'past_key_value_{i}'] = key_value_cache_buffers[i]
            ctx_buffer[f'present_key_value_{i}'] = key_value_cache_buffers[i]
        ctx_buffer['sequence_length'] = sequence_length_buffer * (seq_len +
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

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   res.cpu().numpy(),
                                   atol=1e-1)

        # compare generation
        step = 1
        step1_id = torch.randint(100, (batch_size, 1)).int().cuda()
        gen_input_lengths = ctx_input_lengths.clone()
        gen_masked_tokens = torch.zeros((batch_size, total_length),
                                        dtype=torch.int32,
                                        device='cuda')
        gen_position_ids = torch.ones_like(step1_id).int().cuda() * seq_len
        gen_last_token_ids = torch.zeros_like(gen_input_lengths).int().cuda()

        with torch.no_grad():
            hf_outputs = hf_gpt.forward(
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
            'max_input_length': ctx_max_input_length.shape,
            'cache_indirection': cache_indirections[1].shape,
        }
        step1_buffer = {
            'input_ids': step1_id,
            'input_lengths': gen_input_lengths,
            'masked_tokens': gen_masked_tokens,
            'position_ids': gen_position_ids,
            'last_token_ids': gen_last_token_ids,
            'max_input_length': ctx_max_input_length,
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

    @parameterized.expand([(False)])
    def test_greedy_search_float32(self, use_refit):
        model = 'gpt'
        log_level = 'error'
        dtype = 'float32'
        world_size = 1
        rank = 0

        hidden_act = 'gelu'
        n_layer = 2
        max_new_tokens = 1
        batch_size = 4
        seq_len = 128
        use_plugin = False

        do_sample = False
        early_stoppping = False
        num_beams = 1
        num_beam_groups = 1
        temperature = 1
        top_k = 0
        top_p = 0.0
        length_penalty = 1
        repetition_penalty = 1

        gpt_config, hf_gpt = self._gen_hf_gpt(hidden_act, n_layer,
                                              max_new_tokens, dtype)
        runtime, engine_buffer = self._gen_tensorrt_llm_runtime(
            log_level, dtype, world_size, rank, gpt_config, hf_gpt, model,
            use_plugin, batch_size, seq_len, max_new_tokens, use_refit)

        model_config = ModelConfig(vocab_size=gpt_config.vocab_size,
                                   num_layers=gpt_config.n_layer,
                                   num_heads=gpt_config.n_head,
                                   hidden_size=gpt_config.n_embd,
                                   gpt_attention_plugin=False)

        mapping = tensorrt_llm.Mapping(world_size, rank)
        decoder = tensorrt_llm.runtime.GenerationSession(
            model_config, engine_buffer, mapping)
        pad_token_id = 50256
        eos_token_id = 50257
        sampling_config = SamplingConfig(end_id=eos_token_id,
                                         pad_id=pad_token_id,
                                         num_beams=num_beams,
                                         temperature=temperature,
                                         top_k=top_k,
                                         top_p=top_p,
                                         length_penalty=length_penalty,
                                         repetition_penalty=repetition_penalty)
        input_ids = torch.randint(100, (batch_size, seq_len)).int().cuda()
        input_ids[0][-1] = pad_token_id
        input_ids[1][-3:] = pad_token_id
        input_ids[2][-5:] = pad_token_id

        input_lengths = torch.ones(
            (batch_size)).type(torch.int32).cuda() * seq_len

        decoder.setup(batch_size,
                      max_input_length=seq_len,
                      max_new_tokens=max_new_tokens)

        output_ids = decoder.decode(input_ids, input_lengths, sampling_config)
        #TODO: change to actual ragged tensor after GPT plugin supports it
        output_ids_x = decoder.decode_ragged(
            RaggedTensor.from_row_lengths(input_ids, input_lengths, 0, False),
            sampling_config)

        # works because all requests in the batch has same
        # TODO: enable this when GPT Plugin attention works
        # output_ids_y = decoder.decode_batch([t[:input_lengths[i]] for i, t in enumerate(torch.split(input_ids, 1, dim=0))], sampling_config)

        torch.cuda.synchronize()
        torch.testing.assert_close(output_ids, output_ids_x)

        res = output_ids.squeeze()
        res = res[:, -max_new_tokens:]

        ref_output_ids = hf_gpt.generate(input_ids,
                                         do_sample=do_sample,
                                         early_stopping=early_stoppping,
                                         num_beams=num_beams,
                                         temperature=temperature,
                                         top_k=top_k,
                                         top_p=top_p,
                                         num_beam_groups=num_beam_groups,
                                         max_new_tokens=max_new_tokens,
                                         length_penalty=length_penalty,
                                         repetition_penalty=repetition_penalty,
                                         pad_token_id=pad_token_id,
                                         eos_token_id=eos_token_id)
        torch.cuda.synchronize()
        ref = ref_output_ids[:, -max_new_tokens:]

        np.testing.assert_allclose(ref.cpu().numpy(), res.cpu().numpy())


if __name__ == '__main__':
    unittest.main()
