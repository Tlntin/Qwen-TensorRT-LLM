import os
import time

import tensorrt as trt
import torch
from allowed_configs import get_model_config
from base_benchmark import BaseBenchmark, get_engine_name, serialize_engine

import tensorrt_llm
from tensorrt_llm.builder import Builder
from tensorrt_llm.layers import PositionEmbeddingType
from tensorrt_llm.models import smooth_quantize, weight_only_quantize
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode


class GPTBenchmark(BaseBenchmark):

    def __init__(self, engine_dir, model_name, mode, batch_sizes, in_out_lens,
                 dtype, refit, num_beams, top_k, top_p, output_dir):
        super().__init__(engine_dir, model_name, dtype, output_dir)
        self.batch_sizes = batch_sizes
        self.in_out_lens = in_out_lens
        self.refit = refit
        self.num_beams = num_beams
        self.build_time = 0

        if engine_dir is not None:
            # Get build configs from engine directory
            self.use_gpt_attention_plugin = self.config['plugin_config'][
                'gpt_attention_plugin']
            self.num_heads = self.config['builder_config']['num_heads']
            self.hidden_size = self.config['builder_config']['hidden_size']
            self.vocab_size = self.config['builder_config']['vocab_size']
            self.num_layers = self.config['builder_config']['num_layers']
            self.max_batch_size = self.config['builder_config'][
                'max_batch_size']
            self.max_input_len = self.config['builder_config']['max_input_len']
            self.max_output_len = self.config['builder_config'][
                'max_output_len']
            self.multi_query_mode = self.config['builder_config'][
                'multi_query_mode']

            # Deserialize engine from engine directory
            serialize_path = os.path.join(engine_dir, self.engine_name)
            with open(serialize_path, 'rb') as f:
                engine_buffer = f.read()
        else:
            # Build engine
            self.world_size = tensorrt_llm.mpi_world_size()
            self.apply_query_key_layer_scaling = False
            self.use_smooth_quant = False
            self.use_weight_only = False
            self.weight_only_precision = 'int8'
            self.per_token = False
            self.per_channel = False

            self.use_gpt_attention_plugin = False
            self.use_gemm_plugin = False
            self.use_layernorm_plugin = False
            self.use_lookup_plugin = False
            self.enable_context_fmha = True
            self.remove_input_padding = False
            self.multi_query_mode = False
            if mode == 'plugin':
                self.use_gpt_attention_plugin = 'float16'
                self.use_gemm_plugin = 'float16'
                self.use_layernorm_plugin = 'float16'
                self.use_lookup_plugin = 'float16'
            for key, value in get_model_config(model_name).items():
                setattr(self, key, value)

            if self.use_smooth_quant:
                self.quant_mode = QuantMode.use_smooth_quant(
                    self.per_token, self.per_channel)
            elif self.use_weight_only:
                self.quant_mode = QuantMode.use_weight_only(
                    self.weight_only_precision == 'int4')
            else:
                self.quant_mode = QuantMode(0)

            engine_buffer = self.build()

        assert engine_buffer is not None

        model_config = tensorrt_llm.runtime.ModelConfig(
            num_heads=self.num_heads // self.world_size,
            hidden_size=self.hidden_size // self.world_size,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            gpt_attention_plugin=self.use_gpt_attention_plugin,
            multi_query_mode=self.multi_query_mode)
        if model_name == 'chatglm_6b':
            self.sampling_config = tensorrt_llm.runtime.SamplingConfig(
                end_id=130005,
                pad_id=3,
                num_beams=num_beams,
                top_k=top_k,
                top_p=top_p)
            self.decoder = tensorrt_llm.runtime.ChatGLM6BHeadModelGenerationSession(
                model_config, engine_buffer, self.runtime_mapping)
        else:
            self.sampling_config = tensorrt_llm.runtime.SamplingConfig(
                end_id=50256,
                pad_id=50256,
                num_beams=num_beams,
                top_k=top_k,
                top_p=top_p)
            self.decoder = tensorrt_llm.runtime.GenerationSession(
                model_config, engine_buffer, self.runtime_mapping)

    def get_config(self):
        for inlen, outlen in self.in_out_lens:
            if inlen > self.max_input_len or outlen > self.max_output_len:
                print(
                    f'[WARNING] check inlen({inlen}) <= max_inlen({self.max_input_len}) and '
                    f'outlen({outlen}) <= max_outlen({self.max_output_len}) failed, skipping.'
                )
                continue
            for batch_size in self.batch_sizes:
                if batch_size > self.max_batch_size:
                    print(
                        f'[WARNING] check batch_size({batch_size}) '
                        f'<= max_batch_size({self.max_batch_size}) failed, skipping.'
                    )
                    continue
                yield (batch_size, inlen, outlen)

    def prepare_inputs(self, config):
        batch_size, inlen, outlen = config[0], config[1], config[2]
        input_ids = torch.randint(100, (batch_size, inlen)).int().cuda()
        input_lengths = torch.tensor([inlen
                                      for _ in range(batch_size)]).int().cuda()
        return (input_ids, input_lengths)

    def build(self):
        builder = Builder()
        builder_config = builder.create_builder_config(
            name=self.model_name,
            precision=self.dtype,
            timing_cache=None,
            tensor_parallel=self.world_size,  # TP only
            parallel_build=True,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.n_positions,
            apply_query_key_layer_scaling=self.apply_query_key_layer_scaling,
            max_batch_size=self.max_batch_size,
            max_input_len=self.max_input_len,
            max_output_len=self.max_output_len,
            int8=self.quant_mode.has_act_and_weight_quant(),
            use_refit=self.refit,
            opt_level=self.builder_opt,
            multi_query_mode=self.multi_query_mode)
        engine_name = get_engine_name(self.model_name, self.dtype,
                                      self.world_size, self.runtime_rank)

        kv_dtype = trt.float16 if self.dtype == 'float16' else trt.float32

        # Initialize Module
        if self.model_name in [
                'gpt_350m', 'gpt_175b', 'gpt_350m_sq_per_tensor',
                'gpt_350m_sq_per_token_channel'
        ]:
            tensorrt_llm_model = tensorrt_llm.models.GPTLMHeadModel(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                dtype=kv_dtype,
                tensor_parallel=self.world_size,  # TP only
                tensor_parallel_group=list(range(self.world_size)),  # TP only
                apply_query_key_layer_scaling=builder_config.
                apply_query_key_layer_scaling,
                multi_query_mode=self.multi_query_mode)
            if self.use_smooth_quant:
                tensorrt_llm_model = smooth_quantize(tensorrt_llm_model,
                                                     self.quant_mode)
            elif self.use_weight_only and self.weight_only_precision == 'int8':
                tensorrt_llm_model = weight_only_quantize(
                    tensorrt_llm_model, QuantMode.use_weight_only())
            elif self.use_weight_only and self.weight_only_precision == 'int4':
                tensorrt_llm_model = weight_only_quantize(
                    tensorrt_llm_model,
                    QuantMode.use_weight_only(use_int4_weights=True))
        elif self.model_name in ['opt_350m', 'opt_66b']:
            tensorrt_llm_model = tensorrt_llm.models.OPTLMHeadModel(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                dtype=kv_dtype,
                tensor_parallel=self.world_size,  # TP only
                tensor_parallel_group=list(range(self.world_size)),  # TP only
                pre_norm=self.pre_norm,
                do_layer_norm_before=self.do_layer_norm_before)
        elif self.model_name in ['llama_7b', 'llama_30b']:
            tensorrt_llm_model = tensorrt_llm.models.LLaMAForCausalLM(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                dtype=kv_dtype,
                mlp_hidden_size=self.inter_size,
                neox_rotary_style=True,
                tensor_parallel=self.world_size,  # TP only
                tensor_parallel_group=list(range(self.world_size)))
        elif self.model_name in ['gptj_6b']:
            tensorrt_llm_model = tensorrt_llm.models.GPTJForCausalLM(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                rotary_dim=self.rotary_dim,
                dtype=kv_dtype,
                tensor_parallel=self.world_size,  # TP only
                tensor_parallel_group=list(range(self.world_size)),  # TP only
                apply_query_key_layer_scaling=builder_config.
                apply_query_key_layer_scaling)
        elif self.model_name in ['gptneox_20b']:
            tensorrt_llm_model = tensorrt_llm.models.GPTNeoXForCausalLM(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                rotary_dim=self.rotary_dim,
                dtype=kv_dtype,
                tensor_parallel=self.world_size,  # TP only
                tensor_parallel_group=list(range(self.world_size)),  # TP only
                apply_query_key_layer_scaling=builder_config.
                apply_query_key_layer_scaling)
        elif self.model_name in ['chatglm_6b']:
            tensorrt_llm_model = tensorrt_llm.models.ChatGLM6BHeadModel(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                position_embedding_type=PositionEmbeddingType.learned_absolute,
                rotary_embedding_percentage=0.0,
                dtype=kv_dtype,
                tensor_parallel=self.world_size,  # TP only
                tensor_parallel_group=list(range(self.world_size)),  # TP only
                apply_query_key_layer_scaling=builder_config.
                apply_query_key_layer_scaling,
                quant_mode=self.quant_mode,
                multi_query_mode=self.multi_query_mode)
        else:
            raise Exception(f'Unexpected model: {self.model_name}')

        # Module -> Network
        network = builder.create_network()
        network.trt_network.name = engine_name
        if self.use_gpt_attention_plugin:
            network.plugin_config.set_gpt_attention_plugin(
                dtype=self.use_gpt_attention_plugin)
        if self.use_gemm_plugin:
            network.plugin_config.set_gemm_plugin(dtype=self.use_gemm_plugin)
        if self.use_layernorm_plugin:
            network.plugin_config.set_layernorm_plugin(
                dtype=self.use_layernorm_plugin)
        if self.enable_context_fmha:
            network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
        if self.remove_input_padding:
            network.plugin_config.enable_remove_input_padding()

        # Quantization plugins.
        if self.use_smooth_quant:
            network.plugin_config.set_smooth_quant_gemm_plugin(dtype=self.dtype)
            network.plugin_config.set_layernorm_quantization_plugin(
                dtype=self.dtype)
            # FIXME(nkorobov)
            # See https://nvbugs/4164762
            # See https://nvbugs/4174113
            network.plugin_config.set_quantize_tensor_plugin()
            network.plugin_config.set_quantize_per_token_plugin()
        elif self.use_weight_only:
            network.plugin_config.set_weight_only_quant_matmul_plugin(
                dtype='float16')

        if self.world_size > 1:
            network.plugin_config.set_nccl_plugin(self.dtype)

        # Use the plugin for the embedding parallism and sharing
        network.plugin_config.set_lookup_plugin(dtype=self.use_lookup_plugin)

        with net_guard(network):
            # Prepare
            network.set_named_parameters(tensorrt_llm_model.named_parameters())

            # Forward
            inputs = tensorrt_llm_model.prepare_inputs(self.max_batch_size,
                                                       self.max_input_len,
                                                       self.max_output_len,
                                                       True, self.num_beams)
            tensorrt_llm_model(*inputs)

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
        batch_size, inlen, outlen = config[0], config[1], config[2]
        self.decoder.setup(batch_size, inlen, outlen)
        self.decoder.decode(inputs[0], inputs[1], self.sampling_config)
        torch.cuda.synchronize()

    def report(self, config, latency, percentile95, percentile99,
               peak_gpu_used):
        batch_size, inlen, outlen = config[0], config[1], config[2]
        tokens_per_sec = round(batch_size * outlen / (latency / 1000), 2)
        if self.runtime_rank == 0:
            line = '[BENCHMARK] ' + (
                f'model_name {self.model_name} world_size {self.world_size} '
                f'num_heads {self.num_heads} num_layers {self.num_layers} hidden_size {self.hidden_size} '
                f'vocab_size {self.vocab_size} precision {self.dtype} '
                f'batch_size {batch_size} input_length {inlen} output_length {outlen} '
                f'gpu_peak_mem(gb) {peak_gpu_used} build_time(s) {self.build_time} tokens_per_sec {tokens_per_sec} '
                f'percentile95(ms) {percentile95} percentile99(ms) {percentile99} latency(ms) {latency}'
            )
            print(line)
