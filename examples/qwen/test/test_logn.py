import unittest

import numpy as np
import torch
from polygraphy.backend.trt import EngineFromNetwork, TrtRunner, CreateConfig
import tensorrt_llm
from tensorrt_llm import Tensor
import math
import tensorrt as trt
import numpy as np
from parameterized import parameterized
from tensorrt_llm.parameter import Parameter
from tensorrt_llm.functional import (
    Tensor, shape, concat, constant, arange, outer, unary,
    partial, expand, elementwise_binary, shape, pow, cos, sin, slice, expand_dims_like, repeat_interleave, str_dtype_to_trt
)
log = partial(unary, op=trt.UnaryOperation.LOG)
ceil = partial(unary, op=trt.UnaryOperation.CEIL)
div = partial(elementwise_binary, op=trt.ElementWiseOperation.DIV)


class MyLogn(tensorrt_llm.Module):
    def __init__(self, dtype, seq_length, head_size, per_head_dim) -> None:
        super().__init__()
        self.dtype = dtype
        self.seq_length = seq_length
        self.head_size = head_size
        self.per_head_dim = per_head_dim
        logn_array = np.array([
                np.log(i) / np.log(self.seq_length) if i > self.seq_length else 1
                for i in range(1, 32768)
            ],
            dtype=np.float32
        ).reshape(1, -1, 1, 1)
        self.logn_tensor = Parameter(
            value=logn_array,
            dtype=trt.float32,
            shape=[1, 32767, 1, 1],
        )
    
    def forward(self, key, query):
        seq_start = slice(shape(key), [1], [1]) - slice(shape(query), [1], [1])
        seq_end = slice(shape(key), [1], [1])

        logn_shape = self.logn_tensor.value.shape
        logn_tensor = slice(
            input=self.logn_tensor.value,
            starts=concat([0, seq_start, 0, 0]),
            sizes=concat([logn_shape[0], seq_end - seq_start, logn_shape[2], logn_shape[3]]),
        )
        # logn_tensor2 = repeat_interleave(logn_tensor, self.head_size, 2)
        # logn_tensor2 = repeat_interleave(logn_tensor2, self.per_head_dim, 3)
        logn_tensor2 = expand(
            logn_tensor,
            concat([logn_shape[0], seq_end - seq_start, self.head_size, self.per_head_dim])
        )
        query2 = query.cast(trt.float32) * logn_tensor2
        query2 = query2.cast(self.dtype)
        return [logn_tensor2, query2]




class TestFunctional(unittest.TestCase):

    head_size = 16
    per_head_dim = 128
    seq_length = 8192
    base = 10000.0
    dtype = 'float16'

    
    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([('float32', 9886), ('float32', 1886), ("float16", 9886), ("float16", 1886)])
    def test_case(self, dtype, input_length):
        self.dtype = dtype
        batch_size = 1
        # input_seq_len = 13727
        input_seq_len = input_length
        print("\ndtype", dtype, "input_length", input_length)
        if dtype == "float32":
            pt_key = torch.rand(
                [batch_size, input_seq_len, self.head_size, self.per_head_dim],
                dtype=torch.float32
            )
            pt_query = torch.rand(
                [batch_size, input_seq_len, self.head_size, self.per_head_dim],
                dtype=torch.float32
            )
        else:
            pt_key = torch.rand(
                [batch_size, input_seq_len, self.head_size, self.per_head_dim],
                dtype=torch.float16
            )
            pt_query = torch.rand(
                [batch_size, input_seq_len, self.head_size, self.per_head_dim],
                dtype=torch.float16
            )
        

        def test_trt(feed_dict: dict):
            builder = tensorrt_llm.Builder()
            net = builder.create_network()
            with tensorrt_llm.net_guard(net):
                key = Tensor(name='key',
                           shape=pt_key.shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(self.dtype))

                query = Tensor(name='query',
                           shape=pt_query.shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(self.dtype))
                model = MyLogn(
                    dtype=dtype,
                    seq_length=self.seq_length,
                    head_size=self.head_size,
                    per_head_dim=self.per_head_dim,
                )
                outputs = model.forward(query=query, key=key)
                net._mark_output(outputs[0], 'logn', str_dtype_to_trt(dtype))
                net._mark_output(outputs[1], 'query_output', str_dtype_to_trt(dtype))
                # net._mark_output(outputs[0], 'logn', trt.float32)
                # net._mark_output(outputs[1], 'query_output', trt.float32)

                for k, v in model.named_network_outputs():
                    net._mark_output(v, k, tensorrt_llm.str_dtype_to_trt(dtype))
                    # net._mark_output(v, k, trt.float32)
            # for new
            build_engine = EngineFromNetwork(
                    (builder.trt_builder, net.trt_network),
                    config=CreateConfig(
                        fp16=(dtype == 'float16'),
                        precision_constraints="obey",
                    )
                )
            with TrtRunner(build_engine) as runner:
                outputs = runner.infer(feed_dict=feed_dict)
                # {"key": pt_key.numpy(), "query": pt_query.numpy()}
                return outputs
        
        def test_pytorch(pt_query, pt_key):
            # torch impl
            pt_logn_list = [
                math.log(i, self.seq_length) if i > self.seq_length else 1
                for i in range(1, 32768)
            ]
            pt_logn_tensor = torch.tensor(pt_logn_list, dtype=torch.float32)[None, :, None, None]
            pt_seq_start = pt_key.size(1) - pt_query.size(1)
            pt_seq_end = pt_key.size(1)
            pt_logn_tensor = pt_logn_tensor[:, pt_seq_start: pt_seq_end, :, :].type_as(pt_query)
            pt_logn_tensor2 = pt_logn_tensor.expand_as(pt_query)
            pt_logn_tensor2 = pt_logn_tensor2.to(torch.float32)
            raw_type = pt_query.dtype
            pt_query2 = pt_query.to(torch.float32) * pt_logn_tensor2
            pt_logn_tensor2 = pt_logn_tensor2.to(raw_type)
            pt_query2 = pt_query2.to(raw_type)
            print(
                "pt_logn2 shpae/mean/sum/dtype",
                pt_logn_tensor2.shape,
                pt_logn_tensor2.to(torch.float32).mean().item(),
                pt_logn_tensor2.to(torch.float32).sum().item(),
                pt_logn_tensor2.dtype
            )
            print(
                "pt_query2 shpae/mean/sum/dtype",
                pt_query2.shape,
                pt_query2.to(torch.float32).mean(),
                pt_query2.to(torch.float32).sum(),
                pt_query2.dtype
            )
            return [pt_logn_tensor2, pt_query2]
        
        
        (pt_logn2, pt_query2) = test_pytorch(pt_query=pt_query, pt_key=pt_key)
        outputs = test_trt(feed_dict={"key": pt_key.numpy(), "query": pt_query.numpy()})
        rtol = atol = 1e-9
        print(
            "logn shpae/mean/sum/dtype",
            outputs['logn'].shape,
            outputs['logn'].astype(np.float32).mean(),
            outputs['logn'].astype(np.float32).sum(),
            outputs['logn'].dtype
        )
        print(
            "query_output shpae/mean/sum/dtype",
            outputs['query_output'].shape,
            outputs['query_output'].astype(np.float32).mean(),
            outputs['query_output'].astype(np.float32).sum(),
            outputs['query_output'].dtype
        )
        np.testing.assert_allclose(pt_logn2.cpu().numpy(), outputs['logn'], rtol=rtol, atol=atol)
        np.testing.assert_allclose(pt_query2.cpu().numpy(), outputs['query_output'], rtol=rtol, atol=atol)

if __name__ == "__main__":
    unittest.main()