import math
import unittest
from itertools import product

import numpy as np
import tensorrt as trt
import torch
from parameterized import parameterized
from transformers import GPT2Config, GPTBigCodeConfig, GPTJConfig, LlamaConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.gpt_bigcode.modeling_gpt_bigcode import \
    GPTBigCodeAttention
from transformers.models.gptj.modeling_gptj import GPTJAttention
from transformers.models.llama.modeling_llama import (LlamaAttention,
                                                      _expand_mask,
                                                      _make_causal_mask)

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.runtime import GenerationSequence, KVCacheManager


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def _build_trt_engine(self, trt_network, trt_builder, dtype, shape_dict,
                          use_int8):
        config = trt_builder.create_builder_config()
        if dtype == 'float16':
            config.flags = 1 << (int)(trt.BuilderFlag.FP16)

        opt_profile = trt_builder.create_optimization_profile()
        # Set optimization profiles for the input bindings that need them
        for i in range(trt_network.num_inputs):
            inp_tensor = trt_network.get_input(i)
            name = inp_tensor.name
            # Set profiles for dynamic execution tensors
            if not inp_tensor.is_shape_tensor and -1 in inp_tensor.shape:
                dims = trt.Dims(shape_dict[name])
                opt_profile.set_shape(name, dims, dims, dims)
        config.add_optimization_profile(opt_profile)
        return trt_builder.build_engine(trt_network, config)

    def load_test_cases():
        test_cases = list(
            product(['gpt2_attention', 'llama_attention', 'gptj_attention'],
                    [ContextFMHAType.disabled], ['float16'], [2], [128], [4],
                    [64], [False], [False], [False], [False], [1],
                    [False, True]))

        # TODO: add more unit tests
        test_cases += list(
            product(['llama_attention'], [
                ContextFMHAType.disabled, ContextFMHAType.enabled,
                ContextFMHAType.enabled_with_fp32_acc
            ], ['float16'], [2], [90, 1024], [4], [32, 64, 128], [False],
                    [False], [False], [False], [1], [False]))

        # Test cases for the multi-block MMHA.
        test_cases += list(
            product(['llama_attention'], [
                ContextFMHAType.enabled, ContextFMHAType.enabled_with_fp32_acc
            ], ['float16', 'float32'], [2], [2048], [4], [64], [True], [False],
                    [False], [False], [1], [False]))

        # Test cases for the int8 K/V cache.
        test_cases += list(
            product(['gpt2_attention'], [ContextFMHAType.disabled],
                    ['float16', 'float32'], [2], [128], [4], [64], [False],
                    [False], [True], [False], [1], [False, True]))

        # Test cases for the fp8 K/V cache.
        test_cases += list(
            product(['gpt2_attention'], [ContextFMHAType.disabled],
                    ['float16', 'float32'], [2], [128], [4], [64], [False],
                    [False], [False], [True], [4], [False, True]))

        # test cases for multi-query attention
        test_cases += list(
            product(
                ['gpt2_attention', 'llama_attention', 'gpt_bigcode_attention'],
                [
                    ContextFMHAType.disabled, ContextFMHAType.enabled,
                    ContextFMHAType.enabled_with_fp32_acc
                ], ['float16'], [2], [128], [4], [64], [False], [True], [False],
                [False], [1], [False, True]))

        # test cases for beam search
        test_cases += list(
            product(['gpt2_attention'], [ContextFMHAType.disabled], ['float16'],
                    [2], [128], [4], [64], [False], [False], [False], [False],
                    [4], [False, True]))
        return test_cases

    def custom_name_func(testcase_func, param_num, param):
        return "%s_%s" % (
            testcase_func.__name__,
            parameterized.to_safe_name("_".join(str(x) for x in param.args)),
        )

    @parameterized.expand(load_test_cases, name_func=custom_name_func)
    def test_ib_gpt_attention(self, attention_type, context_fmha_type, dtype,
                              batch_size, in_len, num_heads, head_size,
                              enable_multi_block_mmha,
                              enable_multi_query_attention, use_int8_kv_cache,
                              use_fp8_kv_cache, beam_width, paged_kv_cache):

        session = None

        kv_cache_dtype = 'int8' if use_int8_kv_cache else dtype
        # Fixing seed to avoid flakiness in tests with quantization
        torch.manual_seed(42)

        def _construct_execution(session, input_tensor, weight, bias,
                                 host_beam_widths, host_input_lengths,
                                 input_lengths, past_key_value,
                                 kv_block_pointers, host_past_key_value_lengths,
                                 cache_indirection,
                                 host_req_cache_max_seq_lengths, num_heads,
                                 hidden_size, output, dtype, shape_dict,
                                 enable_multi_query_attention,
                                 kv_8bits_quant_scale, kv_8bits_dequant_scale):
            num_req = host_input_lengths.shape[0]
            max_input_len = host_input_lengths.numpy().max()
            max_beam_width = host_beam_widths.numpy().max()
            head_size = hidden_size // num_heads
            # construct trt network
            builder = tensorrt_llm.Builder()
            net = builder.create_network()
            net.plugin_config.set_inflight_batching_gpt_attention_plugin(dtype)
            net.plugin_config.set_context_fmha(context_fmha_type)
            net.plugin_config.enable_remove_input_padding()
            if paged_kv_cache:
                net.plugin_config.enable_paged_kv_cache()
            # change this to HOST when trt is updated to allow host inputs for plugin layers
            host_loc = trt.TensorLocation.DEVICE
            with tensorrt_llm.net_guard(net):
                network = tensorrt_llm.default_trtnet()
                x_tensor = Tensor(name='input',
                                  shape=tuple(input_tensor.shape),
                                  dtype=tensorrt_llm.str_dtype_to_trt(dtype))
                past_key_value_tensor = Tensor(
                    name='past_key_value',
                    shape=tuple(past_key_value.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt(kv_cache_dtype))
                host_beam_width_tensor = Tensor(
                    name='host_beam_widths',
                    shape=(num_req, ),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'),
                    location=host_loc)
                host_input_length_tensor = Tensor(
                    name='host_input_lengths',
                    shape=(num_req, ),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'),
                    location=host_loc)
                input_lengths_tensor = Tensor(
                    name='input_lengths',
                    shape=tuple(input_lengths.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                past_key_value_ptr_tensor = Tensor(
                    name='past_key_value_pointers',
                    shape=(num_req, 2),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'),
                    location=host_loc)
                host_past_key_value_length_tensor = Tensor(
                    name='host_past_key_value_lengths',
                    shape=tuple(host_past_key_value_lengths.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'),
                    location=host_loc)
                cache_indirection_pointers_tensor = Tensor(
                    name='cache_indirection_pointers',
                    shape=(num_req, 2),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'),
                    location=host_loc)
                host_cache_capacity_tensor = Tensor(
                    name='host_req_cache_max_seq_lengths',
                    shape=(num_req, ),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'),
                    location=host_loc)
                kv_8bits_quant_scale_tensor = None
                kv_8bits_dequant_scale_tensor = None
                if use_int8_kv_cache or use_fp8_kv_cache:
                    kv_8bits_quant_scale_tensor = Tensor(
                        name='kv_8bits_quant_scale',
                        shape=(1, ),
                        dtype=tensorrt_llm.str_dtype_to_trt('float32'))
                    kv_8bits_dequant_scale_tensor = Tensor(
                        name='kv_8bits_dequant_scale',
                        shape=(1, ),
                        dtype=tensorrt_llm.str_dtype_to_trt('float32'))
                pointers_to_kv_block_pointers_tensor = None
                kv_block_pointers_tensor = None
                if paged_kv_cache:
                    pointers_to_kv_block_pointers_tensor = Tensor(
                        name='pointers_to_kv_cache_block_pointers',
                        shape=(num_req, 2),
                        dtype=tensorrt_llm.str_dtype_to_trt('int32'),
                        location=host_loc)
                    kv_block_pointers_tensor = Tensor(
                        name='kv_cache_block_pointers',
                        shape=tuple(kv_block_pointers.shape),
                        dtype=tensorrt_llm.str_dtype_to_trt('int32'))

                linear = tensorrt_llm.layers.Linear(hidden_size,
                                                    weight.size()[-1],
                                                    bias=attention_type in [
                                                        'gpt2_attention',
                                                        'llama_attention',
                                                        'gpt_bigcode_attention'
                                                    ])
                linear.weight.value = np.ascontiguousarray(
                    weight.cpu().numpy().transpose())
                if attention_type in [
                        'gpt2_attention', 'llama_attention',
                        'gpt_bigcode_attention'
                ]:
                    linear.bias.value = bias.cpu().numpy()
                qkv = linear(x_tensor)

                rotary_embedding_dim = head_size if attention_type in [
                    'llama_attention', 'gptj_attention'
                ] else 0
                neox_rotary_style = True if attention_type in [
                    'llama_attention'
                ] else False
                outputs = tensorrt_llm.functional.inflight_batching_gpt_attention(
                    qkv,
                    past_key_value_tensor,
                    host_beam_width_tensor,
                    host_input_length_tensor,
                    input_lengths_tensor,
                    past_key_value_ptr_tensor,
                    host_past_key_value_length_tensor,
                    cache_indirection_pointers_tensor,
                    host_cache_capacity_tensor,
                    num_heads=num_heads,
                    head_size=head_size,
                    q_scaling=1.0,
                    rotary_embedding_dim=rotary_embedding_dim,
                    neox_rotary_style=neox_rotary_style,
                    multi_block_mode=enable_multi_block_mmha,
                    multi_query_mode=enable_multi_query_attention,
                    kv_orig_quant_scale=kv_8bits_quant_scale_tensor,
                    kv_quant_orig_scale=kv_8bits_dequant_scale_tensor,
                    use_int8_kv_cache=use_int8_kv_cache,
                    use_fp8_kv_cache=use_fp8_kv_cache,
                    max_input_len=max_input_len,
                    max_beam_width=max_beam_width,
                    pointers_to_kv_cache_block_pointers=
                    pointers_to_kv_block_pointers_tensor,
                    kv_cache_block_pointers=kv_block_pointers_tensor)

                outputs[0].trt_tensor.name = 'output'
                outputs[0].trt_tensor.dtype = tensorrt_llm.str_dtype_to_trt(
                    dtype)
                network.mark_output(outputs[0].trt_tensor)
                outputs[1].trt_tensor.name = 'present_key_value'
                outputs[1].trt_tensor.dtype = tensorrt_llm.str_dtype_to_trt(
                    kv_cache_dtype)
                network.mark_output(outputs[1].trt_tensor)

            def collect_pointers(t: torch.Tensor, mask_out=[], leadingDim=1):
                pointers = []
                base = t.data_ptr()
                stride = t.element_size() * np.array(
                    t.shape[1:]).prod() * leadingDim
                to_uint32x2 = lambda p: [
                    np.uint32(p & 0xFFFFFFFF),
                    np.uint32((p >> 32) & 0xFFFFFFFF)
                ]
                for i in range(t.shape[0] // leadingDim):
                    pointers.append(
                        to_uint32x2(0 if i in mask_out else base + stride * i))
                return torch.tensor(np.array(pointers).view(np.int32),
                                    dtype=torch.int32,
                                    device='cpu')

            inputs = {
                'input':
                input_tensor,
                'past_key_value':
                past_key_value,
                'host_beam_widths':
                host_beam_widths,
                'host_input_lengths':
                host_input_lengths,
                'input_lengths':
                input_lengths,
                'past_key_value_pointers':
                collect_pointers(past_key_value, leadingDim=beam_width),
                'host_past_key_value_lengths':
                host_past_key_value_lengths,
                'cache_indirection_pointers':
                collect_pointers(cache_indirection,
                                 mask_out=[
                                     i for i in range(num_req)
                                     if host_past_key_value_lengths[i] == 0
                                 ]),
                'host_req_cache_max_seq_lengths':
                host_req_cache_max_seq_lengths
            }
            if use_int8_kv_cache or use_fp8_kv_cache:
                inputs['kv_8bits_quant_scale'] = kv_8bits_quant_scale
                inputs['kv_8bits_dequant_scale'] = kv_8bits_dequant_scale

            if paged_kv_cache:
                inputs['kv_cache_block_pointers'] = kv_block_pointers
                inputs[
                    'pointers_to_kv_cache_block_pointers'] = collect_pointers(
                        kv_block_pointers)

            outputs = {
                'output': output,
                'present_key_value': past_key_value,
            }

            stream = torch.cuda.current_stream()
            builder_config = builder.create_builder_config(
                name=attention_type, precision=dtype, int8=use_int8_kv_cache)
            if session is None:
                engine = builder.build_engine(net, builder_config)
                session = tensorrt_llm.runtime.Session.from_serialized_engine(
                    engine)
            session.run(inputs=inputs,
                        outputs=outputs,
                        stream=stream.cuda_stream)

            torch.cuda.synchronize()

            return session, outputs['output'], outputs['present_key_value']

        hidden_size = num_heads * head_size  # embed dimension
        # If enable_multi_query_attention is true and that GPTBigCodeAttention is tested, use compacted IO shape.
        # If enable_multi_query_attention is true but other attention types are tested, use regular IO shape.
        # This is because GPTBigCodeAttention requires single KV head when multi-query attention is used. Other attention types do not support
        # single KV head natively so we emulate the effect of multi-query attention by repeating KV heads.
        kv_num_heads = 1 if enable_multi_query_attention and attention_type == 'gpt_bigcode_attention' else num_heads
        qkv_hidden_size = hidden_size + 2 * kv_num_heads * head_size
        out_len = 8
        max_seq_len = in_len + 24
        num_req = batch_size
        tokens_per_block = 8
        max_blocks_per_seq = math.ceil(max_seq_len / tokens_per_block)
        blocks = math.ceil(
            (num_req * beam_width * max_seq_len) / tokens_per_block)
        shape_dict = {
            'weight': (hidden_size, qkv_hidden_size),
            'bias': (qkv_hidden_size, ),
            'kv_8bits_quant_scale': (1, ),
            'kv_8bits_dequant_scale': (1, ),
            'cache_indirection': (num_req, beam_width, max_seq_len)
        }
        if paged_kv_cache:
            shape_dict['past_key_value'] = (blocks, 2, kv_num_heads,
                                            tokens_per_block, head_size)
        else:
            shape_dict['past_key_value'] = (num_req * beam_width, 2,
                                            kv_num_heads, max_seq_len,
                                            head_size)
        shape_dict['present_key_value'] = shape_dict['past_key_value']

        present_key_value = torch.zeros(
            shape_dict['present_key_value'],
            dtype=tensorrt_llm._utils.str_dtype_to_torch(kv_cache_dtype),
            device='cuda')

        weight = torch.randn(
            shape_dict['weight'],
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device='cuda') * 1e-3
        # FIXME(qijun): test_gpt_attention_llama_attention_False_float16_2_90_4_64_False_False_False_True
        # fails with xavier_uniform_ initialization
        # torch.nn.init.xavier_uniform_(weight)

        bias = torch.randn(shape_dict['bias'],
                           dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                           device='cuda') * 1e-2

        cache_indirection = torch.zeros(shape_dict['cache_indirection'],
                                        dtype=torch.int32,
                                        device='cuda')
        for i in range(1, beam_width):
            cache_indirection[:, i, in_len:] = i

        if use_fp8_kv_cache:
            kv_8bits_dequant_scale = torch.ones(
                shape_dict['kv_8bits_dequant_scale'],
                dtype=torch.float32,
                device='cuda')
        else:
            kv_8bits_dequant_scale = torch.randint(
                1,
                10,
                shape_dict['kv_8bits_dequant_scale'],
                dtype=torch.float32,
                device='cuda') * 0.0001
        kv_8bits_quant_scale = 1.0 / kv_8bits_dequant_scale

        ConfigCls = None
        AttentionCls = None
        if attention_type == 'gpt2_attention':
            ConfigCls = GPT2Config
            AttentionCls = GPT2Attention
        elif attention_type == 'gptj_attention':
            ConfigCls = GPTJConfig
            AttentionCls = GPTJAttention
        elif attention_type == 'llama_attention':
            ConfigCls = LlamaConfig
            AttentionCls = LlamaAttention
        elif attention_type == 'gpt_bigcode_attention':
            ConfigCls = GPTBigCodeConfig
            AttentionCls = GPTBigCodeAttention

        configuration = ConfigCls(
            hidden_size=hidden_size,
            num_hidden_layers=1,
            num_attention_heads=num_heads,
            vocab_size=51200,
            use_cache=True,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            hidden_act='gelu',
            torch_dtype=dtype,
        )
        attention = AttentionCls(configuration).cuda().eval()
        if attention_type == 'gpt2_attention':
            attention.c_attn.weight = torch.nn.parameter.Parameter(
                data=weight.clone().detach(), requires_grad=False)
            attention.c_attn.bias = torch.nn.parameter.Parameter(
                data=bias.clone().detach(), requires_grad=False)
            attention.c_proj.weight = torch.nn.parameter.Parameter(
                data=torch.eye(
                    hidden_size,
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                    device='cuda'),
                requires_grad=False)
            attention.c_proj.bias = torch.nn.parameter.Parameter(
                data=torch.zeros(
                    (hidden_size, ),
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                    device='cuda'),
                requires_grad=False)
        elif attention_type == 'llama_attention':
            q_w, k_w, v_w = torch.tensor_split(weight, 3, dim=-1)
            q_b, k_b, v_b = torch.tensor_split(bias, 3)
            attention.q_proj.weight = torch.nn.parameter.Parameter(
                data=q_w.contiguous().clone().detach(), requires_grad=False)
            attention.k_proj.weight = torch.nn.parameter.Parameter(
                data=k_w.contiguous().clone().detach(), requires_grad=False)
            attention.v_proj.weight = torch.nn.parameter.Parameter(
                data=v_w.contiguous().clone().detach(), requires_grad=False)

            attention.q_proj.bias = torch.nn.parameter.Parameter(
                data=q_b.contiguous().clone().detach(), requires_grad=False)
            attention.k_proj.bias = torch.nn.parameter.Parameter(
                data=k_b.contiguous().clone().detach(), requires_grad=False)
            attention.v_proj.bias = torch.nn.parameter.Parameter(
                data=v_b.contiguous().clone().detach(), requires_grad=False)

            attention.o_proj.weight = torch.nn.parameter.Parameter(
                data=torch.eye(
                    hidden_size,
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                    device='cuda'),
                requires_grad=False)
            attention.o_proj.bias = torch.nn.parameter.Parameter(
                data=torch.zeros(
                    (hidden_size, ),
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                    device='cuda'),
                requires_grad=False)
        elif attention_type == 'gptj_attention':
            q_w, k_w, v_w = torch.tensor_split(weight, 3, dim=-1)
            attention.q_proj.weight = torch.nn.parameter.Parameter(
                data=q_w.contiguous().clone().detach(), requires_grad=False)
            attention.k_proj.weight = torch.nn.parameter.Parameter(
                data=k_w.contiguous().clone().detach(), requires_grad=False)
            attention.v_proj.weight = torch.nn.parameter.Parameter(
                data=v_w.contiguous().clone().detach(), requires_grad=False)

            attention.out_proj.weight = torch.nn.parameter.Parameter(
                data=torch.eye(
                    hidden_size,
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                    device='cuda'),
                requires_grad=False)
        elif attention_type == 'gpt_bigcode_attention':
            attention.c_attn.weight = torch.nn.parameter.Parameter(
                data=weight.transpose(0, 1).clone().detach(),
                requires_grad=False)
            attention.c_attn.bias = torch.nn.parameter.Parameter(
                data=bias.clone().detach(), requires_grad=False)
            attention.c_proj.weight = torch.nn.parameter.Parameter(
                data=torch.eye(
                    hidden_size,
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                    device='cuda'),
                requires_grad=False)
            attention.c_proj.bias = torch.nn.parameter.Parameter(
                data=torch.zeros(
                    (hidden_size, ),
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                    device='cuda'),
                requires_grad=False)
            attention.layer_idx = 0
        else:
            raise RuntimeError("attention_type not properly set")

        ctx_attention_mask_list = [None] * num_req

        # Setup weights/biases for MQA: key/value shares weights/biases across heads
        if attention_type != 'gpt_bigcode_attention' and enable_multi_query_attention:
            q_w, k_w, v_w = torch.tensor_split(weight, 3, dim=-1)
            q_b, k_b, v_b = torch.tensor_split(bias, 3)
            k_w_head = k_w[:, :head_size]
            v_w_head = k_w[:, :head_size]
            k_w_repeat = k_w_head.repeat(1, num_heads)
            v_w_repeat = v_w_head.repeat(1, num_heads)
            k_b_head = k_b[:head_size]
            v_b_head = v_b[:head_size]
            k_b_repeat = k_b_head.repeat(num_heads)
            v_b_repeat = v_b_head.repeat(num_heads)

            # Setup MQA weights/biases for _construct_execution()
            weight = torch.cat([q_w, k_w_repeat, v_w_repeat], dim=-1)
            bias = torch.cat([q_b, k_b_repeat, v_b_repeat])

            # Plugin will always use compacted MQA format without repeating KV heads
            weight_plugin = torch.cat([q_w, k_w_head, v_w_head], dim=-1)
            bias_plugin = torch.cat([q_b, k_b_head, v_b_head])

            # Setup MQA weights/biases for torch
            if attention_type == 'gpt2_attention':
                attention.c_attn.weight = torch.nn.parameter.Parameter(
                    data=weight.clone().detach(), requires_grad=False)
                attention.c_attn.bias = torch.nn.parameter.Parameter(
                    data=bias.clone().detach(), requires_grad=False)
            elif attention_type == 'llama_attention':
                attention.k_proj.weight = torch.nn.parameter.Parameter(
                    data=k_w_repeat.contiguous().clone().detach(),
                    requires_grad=False)
                attention.v_proj.weight = torch.nn.parameter.Parameter(
                    data=v_w_repeat.contiguous().clone().detach(),
                    requires_grad=False)
                attention.k_proj.bias = torch.nn.parameter.Parameter(
                    data=k_b_repeat.contiguous().clone().detach(),
                    requires_grad=False)
                attention.v_proj.bias = torch.nn.parameter.Parameter(
                    data=v_b_repeat.contiguous().clone().detach(),
                    requires_grad=False)
            elif attention_type == 'gptj_attention':
                attention.k_proj.weight = torch.nn.parameter.Parameter(
                    data=k_w_repeat.contiguous().clone().detach(),
                    requires_grad=False)
                attention.v_proj.weight = torch.nn.parameter.Parameter(
                    data=v_w_repeat.contiguous().clone().detach(),
                    requires_grad=False)
            else:
                raise RuntimeError("attention_type not properly set")

        else:  # not enable_multi_query_attention
            weight_plugin = weight
            bias_plugin = bias

        # torch execution for one sequence
        def torch_exec(step: int,
                       input: torch.Tensor,
                       ctx_attention_mask: torch.Tensor,
                       layer_past=None):
            assert layer_past != None or input.shape[0] == 1
            nonlocal attention
            nonlocal attention_type
            nonlocal in_len
            position_ids = ctx_attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(ctx_attention_mask == 0, 1)
            if step != 0:
                position_ids = position_ids[:, -1].unsqueeze(-1)

            attention_mask = _expand_mask(
                ctx_attention_mask,
                dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                tgt_len=(in_len if step == 0 else 1))
            if attention_type == 'gpt2_attention':
                torch_output, torch_present = attention(
                    input,
                    layer_past=layer_past,
                    use_cache=True,
                    attention_mask=attention_mask)
            elif attention_type == 'llama_attention':
                attention_mask = attention_mask + _make_causal_mask(
                    input.shape[:2],
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                    device='cuda',
                    past_key_values_length=(0 if step == 0 else in_len + step -
                                            1))
                torch_output, _, torch_present = attention(
                    input,
                    past_key_value=layer_past,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=True)
            elif attention_type == 'gptj_attention':
                torch_output, torch_present = attention(
                    input,
                    layer_past=layer_past,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=True)
            elif attention_type == 'gpt_bigcode_attention':
                # source shape = (b, 1, s_query or 1, s_key)
                # target shape = (b, s_query or 1, h, s_key)
                attention_mask = (attention_mask >= 0).permute(
                    [0, 2, 1,
                     3]).expand(input.shape[0], in_len if step == 0 else 1,
                                num_heads, in_len + step)
                torch_output, torch_present = attention(
                    input,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    use_cache=True)
            else:
                raise RuntimeError("attention_type not properly set")

            torch.cuda.synchronize()
            return torch_output, torch_present

        if paged_kv_cache:
            # Init KV cache block manager
            kv_cache_manager = KVCacheManager([present_key_value],
                                              blocks,
                                              tokens_per_block,
                                              max_blocks_per_seq,
                                              beam_width=beam_width)

        torch_cache_list = [None] * batch_size
        sequences = []
        # We don't start all requests together. The i-th request starts from the i-th iteration.
        # Like below, each column is a iteration. c means context step and g means generation step
        # req0 |c|g|g|g|g|g|g|g|
        # req1   |c|g|g|g|g|g|g|g|
        # req2     |c|g|g|g|g|g|g|g|
        # req3       |c|g|g|g|g|g|g|g|
        for i in range(batch_size + out_len - 1):
            pointer_arrays = None
            if paged_kv_cache:
                # Check if new sequence arrived
                if i < batch_size:
                    # Add sequence to the manager
                    sequence = GenerationSequence(seq_idx=i, batch_idx=i)
                    kv_cache_manager.add_sequence(sequence, in_len)

                # Get arrays of pointers to the "pages" of KV values
                pointer_arrays = kv_cache_manager.get_pointer_arrays()[0]

            req_idx_start = max(0, i + 1 - out_len)
            req_idx_end = min(batch_size, i + 1)
            req_idx_list = list(range(req_idx_start, req_idx_end))
            get_step = lambda req_idx: i - req_idx
            step_list = [get_step(req_idx) for req_idx in req_idx_list]
            local_num_req = len(step_list)
            beam_width_list = [
                1 if step == 0 else beam_width for step in step_list
            ]
            host_beam_widths = torch.tensor(beam_width_list,
                                            dtype=torch.int32,
                                            device='cpu')
            input_length_list = [
                in_len if step == 0 else 1 for step in step_list
            ]
            host_input_lengths = torch.tensor(input_length_list,
                                              dtype=torch.int32,
                                              device='cpu')
            input_lengths_replicated = [[in_len] if step == 0 else [1] *
                                        beam_width for step in step_list]
            input_lengths_replicated = [
                x for l in input_lengths_replicated for x in l
            ]
            input_lengths = torch.tensor(input_lengths_replicated,
                                         dtype=torch.int32,
                                         device='cuda')
            total_num_tokens = sum(input_lengths_replicated)
            host_past_key_value_lengths = torch.tensor(
                [0 if step == 0 else in_len + step - 1 for step in step_list],
                dtype=torch.int32,
                device='cpu')

            local_shape_dict = {
                'input': (1, total_num_tokens, hidden_size),
                'output': (1, total_num_tokens, hidden_size),
                'host_beam_widths': (local_num_req, ),
                'host_input_lengths': (local_num_req, ),
                "input_lengths": tuple(input_lengths.shape),
                'past_key_value_ptr': (local_num_req, 2),
                'pointer_array_ptr': (local_num_req, 2),
                'host_past_key_value_lengths': (local_num_req, ),
                'cache_indirection_pointers': (local_num_req, 2),
                'host_req_cache_max_seq_lengths': (local_num_req, ),
                'kv_8bits_quant_scale': (1, ),
                'kv_8bits_dequant_scale': (1, )
            }
            host_req_cache_max_seq_lengths = torch.full((local_num_req, ),
                                                        max_seq_len,
                                                        dtype=torch.int32,
                                                        device='cpu')

            input_tensor = torch.randn(
                local_shape_dict['input'],
                dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                device='cuda') * 1e-3

            output = torch.zeros(
                local_shape_dict['output'],
                dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                device='cuda')

            session, output, present_key_value = _construct_execution(
                session, input_tensor, weight_plugin, bias_plugin,
                host_beam_widths, host_input_lengths, input_lengths,
                present_key_value, pointer_arrays, host_past_key_value_lengths,
                cache_indirection[req_idx_start:req_idx_end, :, :],
                host_req_cache_max_seq_lengths, num_heads, hidden_size, output,
                dtype, local_shape_dict, enable_multi_query_attention,
                kv_8bits_quant_scale, kv_8bits_dequant_scale)

            del session
            session = None

            for req_idx in req_idx_list:
                if get_step(req_idx) == 0:
                    ctx_attention_mask_list[req_idx] = torch.ones(
                        (1, in_len), dtype=torch.int32, device='cuda')
                else:
                    if get_step(req_idx) == 1:
                        ctx_attention_mask_list[req_idx] = torch.ones(
                            (beam_width, in_len),
                            dtype=torch.int32,
                            device='cuda')
                    ctx_attention_mask_list[req_idx] = torch.cat(
                        (ctx_attention_mask_list[req_idx],
                         ctx_attention_mask_list[req_idx].new_ones(
                             (beam_width, 1))),
                        dim=-1).contiguous()

            offset = 0
            for j in range(req_idx_end - req_idx_start):
                req_idx = req_idx_start + j
                step = step_list[j]
                if step == 1 and beam_width > 1:
                    if attention_type != "gpt_bigcode_attention":
                        assert torch_cache_list[req_idx][0].shape[0] == 1
                        torch_cache_list[req_idx] = [
                            x.repeat((beam_width, 1, 1, 1))
                            for x in torch_cache_list[req_idx]
                        ]
                    else:
                        torch_cache_list[req_idx] = torch_cache_list[
                            req_idx].repeat(beam_width, 1, 1)
                input_length = input_length_list[j]
                offset_next = offset + input_length * beam_width
                torch_in = input_tensor[:, offset:offset_next, :].reshape(
                    (beam_width if step != 0 else 1, input_length, hidden_size))
                torch_out, torch_cache_list[req_idx] = torch_exec(
                    step, torch_in, ctx_attention_mask_list[req_idx],
                    torch_cache_list[req_idx])

                np.testing.assert_allclose(
                    output[:, offset:offset_next, :].reshape(
                        (beam_width if step != 0 else 1, input_length,
                         hidden_size)).cpu().numpy(),
                    torch_out.cpu().numpy(),
                    atol=1e-3 if step == 0 else 2E-3)
                offset = offset_next

            if paged_kv_cache:
                num_sequences = req_idx_end - req_idx_start
                finished = [False for _ in range(num_sequences)]
                for j in range(num_sequences):
                    if i - j + 1 == out_len:
                        finished[j] = True
                # Iterate to the next step. Increase number of tokens for all unfinished sequences
                # And allocate new blocks if needed
                kv_cache_manager.step(finished)


if __name__ == "__main__":
    unittest.main()
