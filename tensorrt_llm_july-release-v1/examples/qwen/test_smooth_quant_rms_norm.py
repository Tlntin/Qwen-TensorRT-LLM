import unittest

import numpy as np
import torch
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner
from transformers.models.llama.modeling_llama import LlamaRMSNorm

import tensorrt_llm
from tensorrt_llm import Parameter, Tensor
# from tensorrt_llm.quantization.functional import smooth_quant_rms_norm
from utils.quantization import smooth_quant_rms_norm_op


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([('float16', False), ('float16', True),
                           ('float32', False), ('float32', True)])
    def test_smooth_quant_rms_norm_plugin(self, dtype, dynamic_act_scaling):
        print("test smooth quant rms norm plugin")
        test_shape = [2, 5, 10, 10]

        x_data = torch.randn(
            *test_shape, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

        m = LlamaRMSNorm(test_shape[-1])  # LlamaRMSNorm only supports last dim

        scale_data = torch.randint(2, 32, (1, ), dtype=torch.float32)

        with torch.no_grad():

            def cast_to_int8_with_sat(tensor):
                return tensor.round().clip(-128, 127).to(dtype=torch.int8)

            # pytorch run
            with torch.no_grad():
                ref = m(x_data).to(dtype=torch.float32)
                if dynamic_act_scaling:
                    abs_max_f, _ = ref.abs().max(dim=-1, keepdim=True)
                    dynamic_scale = abs_max_f / 127.0
                    ref_quantized = cast_to_int8_with_sat(ref *
                                                          (127.0 / abs_max_f))
                else:
                    ref_quantized = cast_to_int8_with_sat(ref * scale_data)

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        # net.plugin_config.set_rmsnorm_quantization_plugin(dtype)
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            output = smooth_quant_rms_norm_op(
                x,
                dtype,
                test_shape[-1],
                weight=tensorrt_llm.constant(m.weight.detach().cpu().numpy()),
                scale=Parameter(scale_data.cpu().numpy()).value,
                eps=m.variance_epsilon,
                dynamic_act_scaling=dynamic_act_scaling)

            if dynamic_act_scaling:
                output, dynamic_scales = output
                dynamic_scales = dynamic_scales.trt_tensor
                dynamic_scales.name = 'dynamic_scales'
                network.mark_output(dynamic_scales)
                dynamic_scales.dtype = tensorrt_llm.str_dtype_to_trt('float32')

            output = output.trt_tensor
            output.name = 'output'
            network.mark_output(output)
            output.dtype = tensorrt_llm.str_dtype_to_trt('int8')

            # trt run
            build_engine = EngineFromNetwork(
                (builder.trt_builder, net.trt_network),
                config=CreateConfig(int8=True,
                                    fp16=(dtype == 'float16'),
                                    precision_constraints="obey"))
            assert build_engine is not None, "Build engine failed"
            with TrtRunner(build_engine) as runner:
                outputs = runner.infer(feed_dict={'x': x_data.cpu().numpy()})

            # compare diff of quantized output
            # Set absolute tolerance to 1 to mitigate some rounding error
            np.testing.assert_allclose(ref_quantized.cpu().numpy(),
                                       outputs['output'],
                                       atol=1,
                                       rtol=0)

            # compare diff of dynamic activation scales
            if dynamic_act_scaling:
                np.testing.assert_allclose(dynamic_scale.cpu().numpy(),
                                           outputs['dynamic_scales'],
                                           atol=1e-2)
            print("max diff", np.max(np.abs(ref_quantized.cpu().numpy() - outputs["output"])))

    def test_sq_rms_norm_no_plugin(self):
        print("test seq rms norm no plugin")
        # Create builder
        builder = tensorrt_llm.Builder()
        # Create empty network
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            tensorrt_llm.default_trtnet()
            # Get output tensor for SQ gemm
            with self.assertRaisesRegex(AssertionError, 'Unsupported dtype: 0'):
                smooth_quant_rms_norm_op(None, 0, None, None, None, 0)


if __name__ == '__main__':
    unittest.main()
