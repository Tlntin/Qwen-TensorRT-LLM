import unittest

import numpy as np
import torch
from polygraphy.backend.trt import EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def test_interpolate_without_scales_nearest_5d(self):
        # test data
        dtype = 'float32'
        input_shape = (1, 1, 8, 12, 16)
        output_shape = (16, 24, 32)

        input_data = torch.rand(
            input_shape, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        mode = 'nearest'
        # construct trt network
        align_corners_flag = False

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            input = Tensor(name='input',
                           shape=input_shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.interpolate(
                input=input,
                size=output_shape,
                mode=mode,
                align_corners=align_corners_flag,
            ).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'input': input_data.numpy()})

        ref = torch.nn.functional.interpolate(input_data,
                                              size=output_shape,
                                              mode=mode)

        np.testing.assert_allclose(ref.cpu().numpy(), outputs['output'])

    def test_interpolate_without_scales_bilinear_4d_disable_align_corner(self):
        # test data
        dtype = 'float32'
        input_shape = (1, 1, 8, 12)
        output_shape = (16, 24)

        input_data = torch.rand(
            input_shape, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        mode = 'bilinear'
        # construct trt network
        align_corners_flag = False

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            input = Tensor(name='input',
                           shape=input_shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.interpolate(
                input=input,
                size=output_shape,
                mode=mode,
                align_corners=align_corners_flag,
            ).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'input': input_data.numpy()})

        ref = torch.nn.functional.interpolate(input_data,
                                              size=output_shape,
                                              mode=mode)

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-5)

    def test_interpolate_without_scales_bilinear_4d_enable_align_corner(self):
        # test data
        dtype = 'float32'
        input_shape = (1, 1, 8, 12)
        output_shape = (16, 24)

        input_data = torch.rand(
            input_shape, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        mode = 'bilinear'
        # construct trt network
        align_corners_flag = True

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            input = Tensor(name='input',
                           shape=input_shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.interpolate(
                input=input,
                size=output_shape,
                mode=mode,
                align_corners=align_corners_flag,
            ).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'input': input_data.numpy()})

        ref = torch.nn.functional.interpolate(input_data,
                                              size=output_shape,
                                              mode=mode)

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-5)

    def test_interpolate_without_scales_bicubic_4d_enable_align_corner(self):
        # test data
        dtype = 'float32'
        input_shape = (1, 4, 8, 12)
        output_shape = (16, 24)

        input_data = torch.rand(
            input_shape, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        mode = 'bicubic'
        # construct trt network
        align_corners_flag = True

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            input = Tensor(name='input',
                           shape=input_shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.interpolate(
                input=input,
                size=output_shape,
                mode=mode,
                align_corners=align_corners_flag,
            ).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'input': input_data.numpy()})

        ref = torch.nn.functional.interpolate(input_data,
                                              size=output_shape,
                                              mode=mode)

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-3)

    def test_interpolate_with_scale_3d_nearest_exact(self):
        # test data
        dtype = 'float32'
        input_shape = (1, 4, 8, 16)
        scales_factor = (2, 4)
        input_data = torch.rand(
            input_shape, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        mode = 'nearest-exact'
        # construct trt network
        align_corners_flag = False

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            input = Tensor(name='input',
                           shape=input_shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.interpolate(
                input=input,
                scale_factor=scales_factor,
                mode=mode,
                align_corners=align_corners_flag,
            ).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'input': input_data.numpy()})

        ref = torch.nn.functional.interpolate(input_data,
                                              scale_factor=scales_factor,
                                              mode=mode)
        np.testing.assert_allclose(ref.cpu().numpy(), outputs['output'])

    def test_interpolate_with_scale_4d_bicubic(self):
        # test data
        dtype = 'float32'
        input_shape = (1, 4, 8, 12)
        scales_factor = (2.5, 2)
        input_data = torch.rand(
            input_shape, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        mode = 'bicubic'
        # construct trt network
        align_corners_flag = False

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            input = Tensor(name='input',
                           shape=input_shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.interpolate(
                input=input,
                scale_factor=scales_factor,
                mode=mode,
                align_corners=align_corners_flag,
            ).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'input': input_data.numpy()})

        ref = torch.nn.functional.interpolate(input_data,
                                              scale_factor=scales_factor,
                                              mode=mode)
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-5)

    def test_interpolate_with_scale_4d_bilinear(self):
        # test data
        dtype = 'float32'
        input_shape = (1, 1, 8, 32)
        scales_factor = (2.5, 4)
        input_data = torch.rand(
            input_shape, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        mode = 'bilinear'
        # construct trt network
        align_corners_flag = False

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            input = Tensor(name='input',
                           shape=input_shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.interpolate(
                input=input,
                scale_factor=scales_factor,
                mode=mode,
                align_corners=align_corners_flag,
            ).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'input': input_data.numpy()})

        ref = torch.nn.functional.interpolate(input_data,
                                              scale_factor=scales_factor,
                                              align_corners=align_corners_flag,
                                              mode=mode)
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-5)

    def test_interpolate_with_scale_5d_trilinear_enable_align_corner(self):
        # test data
        dtype = 'float32'
        input_shape = (1, 1, 8, 16, 32)
        scales_factor = (2.5, 2, 4)
        input_data = torch.rand(
            input_shape, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        mode = 'trilinear'
        # construct trt network
        align_corners_flag = True

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            input = Tensor(name='input',
                           shape=input_shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.interpolate(
                input=input,
                scale_factor=scales_factor,
                mode=mode,
                align_corners=align_corners_flag,
            ).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'input': input_data.numpy()})

        ref = torch.nn.functional.interpolate(input_data,
                                              scale_factor=scales_factor,
                                              align_corners=align_corners_flag,
                                              mode=mode)
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-5)
