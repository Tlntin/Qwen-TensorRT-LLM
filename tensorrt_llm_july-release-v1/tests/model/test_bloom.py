import os
import sys
import tempfile
import unittest
from itertools import product

import numpy as np
import tensorrt as trt
import torch
from parameterized import parameterized
from transformers import BloomConfig, BloomForCausalLM

import tensorrt_llm
from tensorrt_llm import Builder
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.network import net_guard
from tensorrt_llm.runtime import ModelConfig, RaggedTensor, SamplingConfig
from tensorrt_llm.runtime.generation import _prepare_attention_mask

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from examples.bloom.weight import load_from_hf_bloom


class TestBloom(unittest.TestCase):

    def _gen_hf_bloom(self, hidden_act, n_layer, max_length, dtype):
        bloom_config = BloomConfig(
            hidden_act=hidden_act,
            n_layer=n_layer,
            max_length=max_length,
            torch_dtype=dtype,
        )

        hf_bloom = BloomForCausalLM(bloom_config).cuda().eval()
        return bloom_config, hf_bloom

    def _gen_tensorrt_llm_network(self, network, builder, hf_bloom,
                                  bloom_config, batch_size, input_len,
                                  output_len, fp16, gpt_attention_plugin,
                                  tensor_parallel,
                                  apply_query_key_layer_scaling):
        num_layers = bloom_config.n_layer
        num_heads = bloom_config.n_head
        hidden_size = bloom_config.hidden_size
        vocab_size = bloom_config.vocab_size
        n_positions = input_len + output_len
        tensor_parallel_group = list(range(tensor_parallel))

        with net_guard(network):
            kv_dtype = trt.float16 if fp16 else trt.float32
            # Initialize model
            tensorrt_llm_bloom = tensorrt_llm.models.BloomForCausalLM(
                num_layers=num_layers,
                num_heads=num_heads,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                hidden_act='gelu',
                max_position_embeddings=n_positions,
                dtype=kv_dtype,
                tensor_parallel=tensor_parallel,  # TP only
                tensor_parallel_group=tensor_parallel_group  # TP only
            )
            inputs = tensorrt_llm_bloom.prepare_inputs(batch_size,
                                                       input_len,
                                                       output_len,
                                                       use_cache=True,
                                                       max_beam_width=1)
            load_from_hf_bloom(tensorrt_llm_bloom, hf_bloom, fp16=fp16)

            # Prepare
            network.set_named_parameters(tensorrt_llm_bloom.named_parameters())

            tensorrt_llm_bloom(*inputs)

        return network

    def _gen_tensorrt_llm_runtime(self,
                                  log_level,
                                  dtype,
                                  world_size,
                                  rank,
                                  bloom_config,
                                  hf_bloom,
                                  model,
                                  use_plugin,
                                  batch_size,
                                  input_len,
                                  output_len,
                                  use_refit,
                                  fast_building=False,
                                  apply_query_key_layer_scaling=False,
                                  enable_context_fmha=False,
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
            if enable_context_fmha:
                network.plugin_config.enable_context_fmha()
            if enable_remove_input_padding:
                network.plugin_config.enable_remove_input_padding()

            self._gen_tensorrt_llm_network(network, builder, hf_bloom,
                                           bloom_config, batch_size, input_len,
                                           output_len, fp16, use_plugin,
                                           world_size,
                                           apply_query_key_layer_scaling)

            builder_config = builder.create_builder_config(
                name='bloom',
                precision=dtype,
                timing_cache='model.cache',
                tensor_parallel=world_size,  # TP only
                use_refit=use_refit,
            )
            engine_buffer = builder.build_engine(network, builder_config)
            runtime = tensorrt_llm.runtime.generation._Runtime(
                engine_buffer, mapping)
        return runtime, engine_buffer

    def load_test_cases():
        # test_cases = list(
        #     product([False, True], [False, True], [False, True], [False, True], ['float16', 'float32']))
        test_cases = list(
            product([False], [True], [False], [False], ['float16', 'float32']))
        return test_cases

    @parameterized.expand(load_test_cases())
    def test_bloom(self, use_refit, use_fast_building, enable_context_fmha,
                   enable_remove_input_padding, dtype):
        model = 'bloom'
        log_level = 'error'
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

        bloom_config, hf_bloom = self._gen_hf_bloom(hidden_act, n_layer,
                                                    max_length, dtype)
        runtime, _ = self._gen_tensorrt_llm_runtime(
            log_level,
            dtype,
            world_size,
            rank,
            bloom_config,
            hf_bloom,
            model,
            use_plugin,
            batch_size,
            seq_len,
            max_length,
            use_refit,
            fast_building=use_fast_building,
            enable_context_fmha=enable_context_fmha,
            enable_remove_input_padding=enable_remove_input_padding)

        # compare context
        pad_token_id = 3
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
        for i in range(bloom_config.n_layer):
            shape = (batch_size, 2, bloom_config.n_head, 0,
                     bloom_config.hidden_size // bloom_config.n_head)
            past_buffer = torch.zeros((1, ),
                                      dtype=str_dtype_to_torch(dtype),
                                      device='cuda')
            ctx_shape.update({
                f'past_key_value_{i}': shape,
            })
            shape = (batch_size, 2, bloom_config.n_head, seq_len,
                     bloom_config.hidden_size // bloom_config.n_head)
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
            hf_outputs = hf_bloom.forward(ctx_ids,
                                          attention_mask=ctx_attention_mask)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   res.cpu().numpy(),
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
        for i in range(bloom_config.n_layer):
            shape = (batch_size, 2, bloom_config.n_head, seq_len,
                     bloom_config.hidden_size // bloom_config.n_head)
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
            hf_outputs = hf_bloom.forward(
                gen_id,
                attention_mask=gen_attention_mask,
                past_key_values=hf_outputs.past_key_values,
                use_cache=True)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   res.cpu().numpy(),
                                   atol=1e-2)

    @parameterized.expand(load_test_cases())
    def test_greedy_search(self, use_refit, use_fast_building,
                           enable_context_fmha, enable_remove_input_padding,
                           dtype):
        model = 'bloom'
        log_level = 'error'
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

        bloom_config, hf_bloom = self._gen_hf_bloom(hidden_act, n_layer,
                                                    max_new_tokens, dtype)
        runtime, engine_buffer = self._gen_tensorrt_llm_runtime(
            log_level,
            dtype,
            world_size,
            rank,
            bloom_config,
            hf_bloom,
            model,
            use_plugin,
            batch_size,
            seq_len,
            max_new_tokens,
            use_refit,
            fast_building=use_fast_building,
            enable_context_fmha=enable_context_fmha,
            enable_remove_input_padding=enable_remove_input_padding)

        model_config = ModelConfig(vocab_size=bloom_config.vocab_size,
                                   num_layers=bloom_config.n_layer,
                                   num_heads=bloom_config.n_head,
                                   hidden_size=bloom_config.hidden_size,
                                   gpt_attention_plugin=False)

        mapping = tensorrt_llm.Mapping(world_size, rank)
        decoder = tensorrt_llm.runtime.GenerationSession(
            model_config, engine_buffer, mapping)
        pad_token_id = 3
        eos_token_id = 2
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
        # TODO: change to actual ragged tensor after BLOOM plugin supports it
        output_ids_x = decoder.decode_ragged(
            RaggedTensor.from_row_lengths(input_ids, input_lengths, 0, False),
            sampling_config)

        torch.cuda.synchronize()
        torch.testing.assert_close(output_ids, output_ids_x)

        res = output_ids.squeeze()
        res = res[:, -max_new_tokens:]

        ref_output_ids = hf_bloom.generate(
            input_ids,
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
