import os
import time
from collections import OrderedDict

import tensorrt as trt
import torch
from allowed_configs import get_model_config
from base_benchmark import BaseBenchmark, serialize_engine

import tensorrt_llm
from tensorrt_llm._utils import trt_dtype_to_torch
from tensorrt_llm.builder import Builder
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.runtime import TensorInfo


class BERTBenchmark(BaseBenchmark):

    def __init__(self, engine_dir, model_name, mode, batch_sizes, in_lens,
                 dtype, output_dir):
        super().__init__(engine_dir, model_name, dtype, output_dir)
        self.batch_sizes = batch_sizes
        self.in_lens = in_lens
        self.build_time = 0

        if engine_dir is not None:
            # Get build configs from engine directory
            self.max_batch_size = self.config['builder_config'][
                'max_batch_size']
            self.max_input_len = self.config['builder_config']['max_input_len']

            # Deserialize engine from engine directory
            serialize_path = os.path.join(engine_dir, self.engine_name)
            with open(serialize_path, 'rb') as f:
                engine_buffer = f.read()
        else:
            # Build engine
            self.use_bert_attention_plugin = False
            self.use_gemm_plugin = False
            self.use_layernorm_plugin = False
            self.enable_qk_half_accum = False
            self.enable_context_fmha = False
            if mode == 'plugin':
                self.use_bert_attention_plugin = 'float16'
                self.use_gemm_plugin = 'float16'
                self.use_layernorm_plugin = 'float16'
            for key, value in get_model_config(model_name).items():
                setattr(self, key, value)

            engine_buffer = self.build()

        assert engine_buffer is not None

        self.session = tensorrt_llm.runtime.Session.from_serialized_engine(
            engine_buffer)

    def get_config(self):
        for inlen in self.in_lens:
            if inlen > self.max_input_len:
                continue
            for batch_size in self.batch_sizes:
                if batch_size > self.max_batch_size:
                    continue
                yield (batch_size, inlen)

    def prepare_inputs(self, config):
        batch_size, inlen = config[0], config[1]
        input_ids = torch.randint(100, (batch_size, inlen)).int().cuda()
        input_lengths = inlen * torch.ones(
            (batch_size, ), dtype=torch.int32, device='cuda')
        inputs = {'input_ids': input_ids, 'input_lengths': input_lengths}
        output_info = self.session.infer_shapes([
            TensorInfo('input_ids', trt.DataType.INT32, input_ids.shape),
            TensorInfo('input_lengths', trt.DataType.INT32, input_lengths.shape)
        ])
        outputs = {
            t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device='cuda')
            for t in output_info
        }
        stream = torch.cuda.current_stream().cuda_stream
        return (inputs, outputs, stream)

    def build(self):
        bs_range = [1, (self.max_batch_size + 1) // 2, self.max_batch_size]
        inlen_range = [1, (self.max_input_len + 1) // 2, self.max_input_len]

        builder = Builder()
        builder_config = builder.create_builder_config(
            name='bert',
            precision=self.dtype,
            timing_cache=None,
            tensor_parallel=self.world_size,  # TP only
            max_batch_size=self.max_batch_size,
            max_input_len=self.max_input_len,
        )
        # Initialize model
        tensorrt_llm_bert = tensorrt_llm.models.BertModel(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.n_positions,
            type_vocab_size=self.type_vocab_size,
            tensor_parallel=self.world_size,  # TP only
            tensor_parallel_group=list(range(self.world_size)))

        # Module -> Network
        network = builder.create_network()
        if self.use_bert_attention_plugin:
            network.plugin_config.set_bert_attention_plugin(
                dtype=self.use_bert_attention_plugin)
        if self.use_gemm_plugin:
            network.plugin_config.set_gemm_plugin(dtype=self.use_gemm_plugin)
        if self.use_layernorm_plugin:
            network.plugin_config.set_layernorm_plugin(
                dtype=self.use_layernorm_plugin)
        if self.enable_qk_half_accum:
            network.plugin_config.enable_qk_half_accum()
        if self.enable_context_fmha:
            network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
        if self.world_size > 1:
            network.plugin_config.set_nccl_plugin(self.dtype)
        with net_guard(network):
            # Prepare
            network.set_named_parameters(tensorrt_llm_bert.named_parameters())

            # Forward
            input_ids = tensorrt_llm.Tensor(
                name='input_ids',
                dtype=trt.int32,
                shape=[-1, -1],
                dim_range=OrderedDict([('batch_size', [bs_range]),
                                       ('input_len', [inlen_range])]),
            )
            input_lengths = tensorrt_llm.Tensor(name='input_lengths',
                                                dtype=trt.int32,
                                                shape=[-1],
                                                dim_range=OrderedDict([
                                                    ('batch_size', [bs_range])
                                                ]))
            hidden_states = tensorrt_llm_bert(input_ids=input_ids,
                                              input_lengths=input_lengths)

            # Mark outputs
            hidden_states_dtype = trt.float16 if self.dtype == 'float16' else trt.float32
            hidden_states.mark_output('hidden_states', hidden_states_dtype)

        # Network -> Engine
        start = time.time()
        engine = builder.build_engine(network, builder_config)
        end = time.time()
        self.build_time = round(end - start, 2)

        if self.output_dir is not None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            serialize_path = os.path.join(self.output_dir, self.engine_name)
            serialize_engine(engine, serialize_path)
            if self.runtime_rank == 0:
                config_path = os.path.join(self.output_dir, 'config.json')
                builder_config.plugin_config = network.plugin_config
                builder.save_config(builder_config, config_path)
        return engine

    def run(self, inputs, config):
        ok = self.session.run(*inputs)
        assert ok, "Runtime execution failed"
        torch.cuda.synchronize()

    def report(self, config, latency, percentile95, percentile99,
               peak_gpu_used):
        if self.runtime_rank == 0:
            line = '[BENCHMARK] ' + (
                f'model_name {self.model_name} world_size {self.world_size} precision {self.dtype} '
                f'batch_size {config[0]} input_length {config[1]} gpu_peak_mem(gb) {peak_gpu_used} '
                f'build_time(s) {self.build_time} percentile95(ms) {percentile95} '
                f'percentile99(ms) {percentile99} latency(ms) {latency}')
            print(line)
