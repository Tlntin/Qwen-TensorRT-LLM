import unittest

import numpy as np
import torch
from parameterized import parameterized

import tensorrt_llm  # NOQA

FP8_E4M3_MAX = 448.0


class TestDynamicFP8QuantDequant(unittest.TestCase):

    @parameterized.expand([(torch.float32), (torch.float16), (torch.bfloat16)])
    def test_quantization_activation_scales(self, dtype):
        A = torch.tensor([[1, 2, 3], [2, 4, 6]], dtype=dtype)
        _, s = torch.ops.tensorrt_llm.quantize_e4m3_activation(A)
        s_ref = (torch.max(A, -1)[0].float() / FP8_E4M3_MAX).to(dtype)

        np.testing.assert_allclose(s_ref.float().numpy(),
                                   s.squeeze().float().numpy())

    @parameterized.expand([(torch.float32), (torch.float16), (torch.bfloat16)])
    def test_quantization_weight_scales(self, dtype):
        A = torch.tensor([[1, 2, 3], [2, 4, 6]], dtype=dtype)
        _, s = torch.ops.tensorrt_llm.quantize_e4m3_weight(A)
        s_ref = (torch.max(A, 0)[0].float() / FP8_E4M3_MAX).to(dtype)

        np.testing.assert_allclose(s_ref.float().numpy(),
                                   s.squeeze().float().numpy())

    @parameterized.expand([(torch.float32), (torch.float16), (torch.bfloat16)])
    def test_quantization_per_tensor_scales(self, dtype):
        A = torch.tensor([[1, 2, 3], [2, 4, 6]], dtype=dtype)
        _, s = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(A)
        s_ref = (A.flatten().max().float() / FP8_E4M3_MAX).to(dtype)

        np.testing.assert_allclose(s_ref.float().numpy(),
                                   s.squeeze().float().numpy())

    @parameterized.expand([(torch.float32), (torch.float16), (torch.bfloat16)])
    def test_quantization_dequantization_activation(self, dtype):
        n = 512
        m = 1024
        A = torch.randn((n, m), dtype=dtype)

        assert A.stride() == (m, 1)

        qA, s = torch.ops.tensorrt_llm.quantize_e4m3_activation(A)

        assert qA.shape == A.shape
        assert qA.shape[:-1] == s.shape[:-1]
        assert s.shape[-1] == 1
        assert s.dtype == A.dtype
        assert qA.dtype == torch.int8

        s_ref = (torch.max(A.float().abs(), -1)[0] / FP8_E4M3_MAX).to(dtype)
        np.testing.assert_allclose(s_ref.float().numpy(),
                                   s.squeeze().float().numpy())

        B = torch.ops.tensorrt_llm.dequantize_e4m3_activation(qA, s)

        assert B.shape == A.shape
        assert B.dtype == A.dtype

        np.testing.assert_allclose(A.float().numpy(),
                                   B.float().numpy(),
                                   atol=0.2)

        # testing exact match
        A = torch.randint(0, 8, (n, m), dtype=dtype)

        qA, s = torch.ops.tensorrt_llm.quantize_e4m3_activation(A)
        B = torch.ops.tensorrt_llm.dequantize_e4m3_activation(qA, s)

        np.testing.assert_allclose(A.float().numpy(), B.float().numpy())

    @parameterized.expand([(torch.float32), (torch.float16), (torch.bfloat16)])
    def test_quantization_dequantization_weight(self, dtype):
        n = 512
        m = 1024
        A = torch.randn((n, m), dtype=dtype)

        assert A.stride() == (m, 1)

        qA, s = torch.ops.tensorrt_llm.quantize_e4m3_weight(A)

        assert qA.shape == A.shape
        assert qA.shape[1:] == s.shape[1:]
        assert s.shape[0] == 1

        s_ref = (torch.max(A.float().abs(), 0)[0] / FP8_E4M3_MAX).to(dtype)
        np.testing.assert_allclose(s_ref.float().numpy(),
                                   s.squeeze().float().numpy())

        B = torch.ops.tensorrt_llm.dequantize_e4m3_weight(qA, s)

        np.testing.assert_allclose(A.float().numpy(),
                                   B.float().numpy(),
                                   atol=0.2)

        # testing exact match
        A = torch.randint(0, 8, (n, m), dtype=dtype)

        qA, s = torch.ops.tensorrt_llm.quantize_e4m3_weight(A)
        B = torch.ops.tensorrt_llm.dequantize_e4m3_weight(qA, s)

        np.testing.assert_allclose(A.float().numpy(), B.float().numpy())

    @parameterized.expand([(torch.float32), (torch.float16), (torch.bfloat16)])
    def test_quantization_dequantization_per_tensor(self, dtype):
        n = 512
        m = 1024
        A = torch.randn((n, m), dtype=dtype)

        qA, s = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(A)

        assert qA.shape == A.shape
        assert qA.dim() == s.dim()
        assert s.numel() == 1

        s_ref = (A.flatten().float().abs().max() / FP8_E4M3_MAX).to(dtype)
        np.testing.assert_allclose(s_ref.float().numpy(),
                                   s.squeeze().float().numpy())

        B = torch.ops.tensorrt_llm.dequantize_e4m3_per_tensor(qA, s)

        # per tensor is less accurate than others, so larger atol is used.
        np.testing.assert_allclose(A.float().numpy(),
                                   B.float().numpy(),
                                   atol=0.25)

        # testing exact match
        A = torch.randint(0, 8, (n, m), dtype=dtype)

        qA, s = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(A)
        B = torch.ops.tensorrt_llm.dequantize_e4m3_per_tensor(qA, s)

        np.testing.assert_allclose(A.float().numpy(), B.float().numpy())
