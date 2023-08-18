import unittest

import numpy as np
import torch
from parameterized import parameterized
from transformers import GPTJConfig
from transformers.models.gptj.modeling_gptj import \
    GPTJAttention as HFGPTJAttention

import tensorrt_llm
from tensorrt_llm import RaggedTensor, Tensor
from tensorrt_llm.models.gptj.model import GPTJAttention


class TestAttention(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def _construct_net(self, dtype, n_embd, n_head, n_positions, rotary_dim,
                       hf_attention, input_tensor, past_key_value_tensor,
                       sequence_length_tensor, past_key_value_length_tensor,
                       masked_tokens_tensor, input_lengths_tensor,
                       max_input_length_tensor, cache_indirection_tensor):
        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        net.plugin_config.set_gpt_attention_plugin(dtype)
        with tensorrt_llm.net_guard(net):
            input = Tensor(name='input',
                           shape=input_tensor.shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            past_key_value = Tensor(name='past_key_value',
                                    shape=tuple(past_key_value_tensor.shape),
                                    dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            sequence_length = Tensor(
                name='sequence_length',
                shape=tuple(sequence_length_tensor.shape),
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            past_key_value_length = Tensor(
                name='past_key_value_length',
                shape=tuple(past_key_value_length_tensor.shape),
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            masked_tokens = Tensor(name='masked_tokens',
                                   shape=tuple(masked_tokens_tensor.shape),
                                   dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            input_lengths = Tensor(name='input_lengths',
                                   shape=tuple(input_lengths_tensor.shape),
                                   dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            max_input_length = Tensor(
                name='max_input_length',
                shape=tuple(max_input_length_tensor.shape),
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))

            cache_indirection = Tensor(
                name='cache_indirection',
                shape=tuple(cache_indirection_tensor.shape),
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))

            gm = GPTJAttention(hidden_size=n_embd,
                               num_attention_heads=n_head,
                               rotary_dim=rotary_dim,
                               dtype=dtype,
                               max_position_embeddings=n_positions)

            q_w = hf_attention.q_proj.weight.detach().cpu().numpy()
            k_w = hf_attention.k_proj.weight.detach().cpu().numpy()
            v_w = hf_attention.v_proj.weight.detach().cpu().numpy()
            dense_w = hf_attention.out_proj.weight.detach().cpu().numpy()
            input_ragged = RaggedTensor.from_row_lengths(
                input, input_lengths, max_input_length)
            gm.qkv.weight.value = np.concatenate([q_w, k_w, v_w])
            gm.dense.weight.value = dense_w

            tensorrt_llm_output, tensorrt_llm_present_key_value = gm.forward(
                input_ragged,
                past_key_value=past_key_value,
                sequence_length=sequence_length,
                past_key_value_length=past_key_value_length,
                use_cache=True,
                masked_tokens=masked_tokens,
                cache_indirection=cache_indirection,
            )
            net._mark_output(tensorrt_llm_output.data,
                             'output',
                             dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            net._mark_output(tensorrt_llm_present_key_value,
                             'present_key_value',
                             dtype=tensorrt_llm.str_dtype_to_trt(dtype))
        return builder, net

    @parameterized.expand([('float16', )])
    def test_gptj_attention(self, dtype):
        # test data
        n_embd = 4096
        n_head = 16
        n_positions = 2048
        rotary_dim = 64
        batch_size = 4
        in_len = 128
        max_seq_len = in_len + 2
        head_size = n_embd // n_head

        config = GPTJConfig(
            n_embd=n_embd,
            n_head=n_head,
            rotary_dim=rotary_dim,
            hidden_act='gelu',
            n_positions=n_positions,
            torch_dtype=dtype,
        )
        hf_attention = HFGPTJAttention(config).cuda().to(
            tensorrt_llm._utils.str_dtype_to_torch(dtype)).eval()

        shape_dict = {
            'input': (batch_size, in_len, n_embd),
            'past_key_value': (batch_size, 2, n_head, max_seq_len, head_size),
            'sequence_length': (batch_size, ),
            'past_key_value_length': (2, ),
            'masked_tokens': (batch_size, max_seq_len),
            'input_lengths': (batch_size, ),
            'max_input_length': (in_len, ),
            'output': (batch_size, in_len, n_embd),
            'present_key_value':
            (batch_size, 2, n_head, max_seq_len, head_size),
        }

        input_tensor = torch.randn(
            shape_dict['input'],
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device='cuda') * 1e-3
        position_ids = torch.arange(0,
                                    in_len,
                                    dtype=torch.long,
                                    device=input_tensor.device)

        # pytorch context run
        with torch.no_grad():
            torch_output, (torch_present_key,
                           torch_present_value) = hf_attention(
                               input_tensor,
                               layer_past=None,
                               position_ids=position_ids,
                               use_cache=True)

        # TRT-LLM context run
        sequence_length_tensor = torch.ones(shape_dict['sequence_length'],
                                            dtype=torch.int32,
                                            device='cuda') * (in_len)
        input_lengths_tensor = torch.ones(shape_dict['input_lengths'],
                                          dtype=torch.int32,
                                          device='cuda') * in_len
        past_key_value_length_tensor = torch.tensor([0, 1], dtype=torch.int32)
        max_input_length_tensor = torch.zeros(shape_dict['max_input_length'],
                                              dtype=torch.int32,
                                              device='cuda')
        masked_tokens_tensor = torch.zeros(shape_dict['masked_tokens'],
                                           dtype=torch.int32,
                                           device='cuda')
        cache_indirection_tensor = torch.full((
            batch_size,
            1,
            max_seq_len,
        ),
                                              0,
                                              dtype=torch.int32,
                                              device='cuda')

        past_key_value_tensor = torch.zeros(
            shape_dict['past_key_value'],
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device='cuda')

        builder, net = self._construct_net(
            dtype, n_embd, n_head, n_positions, rotary_dim, hf_attention,
            input_tensor, past_key_value_tensor, sequence_length_tensor,
            past_key_value_length_tensor, masked_tokens_tensor,
            input_lengths_tensor, max_input_length_tensor,
            cache_indirection_tensor)

        builder_config = builder.create_builder_config(name='gpt-j',
                                                       precision=dtype)

        # Build engine
        engine_buffer = builder.build_engine(net, builder_config)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(
            engine_buffer)
        stream = torch.cuda.current_stream().cuda_stream

        outputs = {
            'output':
            torch.empty(shape_dict['input'],
                        dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                        device='cuda'),
            'present_key_value':
            past_key_value_tensor,
        }
        session.run(inputs={
            'input': input_tensor,
            'past_key_value': past_key_value_tensor,
            'sequence_length': sequence_length_tensor,
            'past_key_value_length': past_key_value_length_tensor,
            'masked_tokens': masked_tokens_tensor,
            'input_lengths': input_lengths_tensor,
            'max_input_length': max_input_length_tensor,
            'cache_indirection': cache_indirection_tensor
        },
                    outputs=outputs,
                    stream=stream)
        torch.cuda.synchronize()

        # compare context diff
        np.testing.assert_allclose(torch_output.cpu().numpy(),
                                   outputs['output'].cpu().numpy(),
                                   atol=1e-04,
                                   rtol=1e-04)

        past_key_value_tensor = outputs['present_key_value'].permute(
            1, 0, 2, 3, 4)
        key, value = past_key_value_tensor.chunk(2)

        # TRT-LLM has a special key cache layout:
        # [bs, n_head, head_size/(16/sizeof(T)), max_seq_len, 16/sizeof(T)]
        key = key.to(torch.float32).reshape(batch_size, n_head, head_size // 8,
                                            max_seq_len,
                                            8).permute(0, 1, 3, 2, 4).reshape(
                                                batch_size, n_head, max_seq_len,
                                                head_size)
        value = value.reshape(batch_size, n_head, max_seq_len, head_size)

        np.testing.assert_allclose(torch_present_key.cpu().numpy(),
                                   key[:, :, :in_len, :].cpu().numpy(),
                                   atol=1e-02,
                                   rtol=1e-02)

        np.testing.assert_allclose(torch_present_value.cpu().numpy(),
                                   value[:, :, :in_len, :].cpu().numpy(),
                                   atol=1e-02,
                                   rtol=1e-02)

        # generation test
        step = 1
        shape_dict['input'] = (batch_size, 1, n_embd)

        input_tensor = torch.randn(
            shape_dict['input'],
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device='cuda') * 1e-3

        position_ids = torch.arange(in_len,
                                    in_len + 1,
                                    dtype=torch.long,
                                    device=input_tensor.device)

        # pytorch generation run
        with torch.no_grad():
            torch_output, (torch_present_key,
                           torch_present_value) = hf_attention(
                               input_tensor,
                               layer_past=(torch_present_key,
                                           torch_present_value),
                               position_ids=position_ids,
                               use_cache=True)

        # TRT-LLM generation run
        past_key_value_length_tensor = torch.tensor([in_len + step - 1, 0],
                                                    dtype=torch.int32)

        sequence_length_tensor = torch.ones(shape_dict['sequence_length'],
                                            dtype=torch.int32,
                                            device='cuda') * (in_len + step)

        builder, net = self._construct_net(
            dtype, n_embd, n_head, n_positions, rotary_dim, hf_attention,
            input_tensor, past_key_value_tensor, sequence_length_tensor,
            past_key_value_length_tensor, masked_tokens_tensor,
            input_lengths_tensor, max_input_length_tensor,
            cache_indirection_tensor)

        builder_config = builder.create_builder_config(name='gpt-j',
                                                       precision=dtype)

        # Build engine
        engine_buffer = builder.build_engine(net, builder_config)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(
            engine_buffer)
        stream = torch.cuda.current_stream().cuda_stream

        outputs = {
            'output':
            torch.empty(shape_dict['input'],
                        dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                        device='cuda'),
            'present_key_value':
            past_key_value_tensor,
        }
        session.run(inputs={
            'input': input_tensor,
            'past_key_value': past_key_value_tensor,
            'sequence_length': sequence_length_tensor,
            'past_key_value_length': past_key_value_length_tensor,
            'masked_tokens': masked_tokens_tensor,
            'input_lengths': input_lengths_tensor,
            'max_input_length': max_input_length_tensor,
            'cache_indirection': cache_indirection_tensor
        },
                    outputs=outputs,
                    stream=stream)
        torch.cuda.synchronize()

        # compare generation diff
        np.testing.assert_allclose(torch_output.cpu().numpy(),
                                   outputs['output'].cpu().numpy(),
                                   atol=1e-04,
                                   rtol=1e-04)

        past_key_value_tensor = past_key_value_tensor.permute(1, 0, 2, 3, 4)
        key, value = past_key_value_tensor.chunk(2)

        # TRT-LLM has a special key cache layout:
        # [bs, n_head, head_size/(16/sizeof(T)), max_seq_len, 16/sizeof(T)]
        key = key.to(torch.float32).reshape(batch_size, n_head, head_size // 8,
                                            max_seq_len,
                                            8).permute(0, 1, 3, 2, 4).reshape(
                                                batch_size, n_head, max_seq_len,
                                                head_size)
        value = value.reshape(batch_size, n_head, max_seq_len, head_size)

        np.testing.assert_allclose(torch_present_key.cpu().numpy(),
                                   key[:, :, :in_len + step, :].cpu().numpy(),
                                   atol=1e-02,
                                   rtol=1e-02)

        np.testing.assert_allclose(torch_present_value.cpu().numpy(),
                                   value[:, :, :in_len + step, :].cpu().numpy(),
                                   atol=1e-02,
                                   rtol=1e-02)


if __name__ == "__main__":
    unittest.main()
