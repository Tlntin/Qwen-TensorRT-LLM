import unittest
from collections import OrderedDict
import numpy as np
import torch
from polygraphy.backend.trt import EngineFromNetwork, TrtRunner, CreateConfig, Profile
import tensorrt_llm
from tensorrt_llm import Tensor
import math
import tensorrt as trt
import numpy as np
from tensorrt_llm.layers import Embedding
from tensorrt_llm import str_dtype_to_trt
from parameterized import parameterized
from tensorrt_llm.functional import (
    Tensor, shape, concat, constant, arange, outer, unary,
    partial, expand, elementwise_binary, shape, pow, cos, sin, slice, maximum
)
log = partial(unary, op=trt.UnaryOperation.LOG)
ceil = partial(unary, op=trt.UnaryOperation.CEIL)
div = partial(elementwise_binary, op=trt.ElementWiseOperation.DIV)
gt = partial(elementwise_binary, op=trt.ElementWiseOperation.GREATER)



class RotaryEmbedding(tensorrt_llm.Module):
    def __init__(self, per_head_dim=128, seq_length=8192, base=10000.0) -> None:
        self.per_head_dim = per_head_dim
        self.seq_length = seq_length
        self.base = base
        super().__init__()
        # self.position_embedding_cos = Embedding(
        #     seq_length,
        #     per_head_dim,
        #     dtype=trt.float32
        # )
        # self.position_embedding_sin = Embedding(
        #     seq_length,
        #     per_head_dim,
        #     dtype=trt.float32
        # )

    def forward(self, input_ids):
        # implement for old
        batch_size = shape(input_ids, 0)
        input_len = shape(input_ids, 1)
        # pytorch impl
        # context_value = math.log(true_seq_len / self.seq_length, 2) + 1
        # ntk_alpha = 2 ** math.ceil(context_value) - 1
        # ntk_alpha = max(ntk_alpha, 1)

        # trt impl
        # with tensorrt_llm.precision("float32"):
        context_value = log(input_len.cast(trt.float32) / float(self.seq_length)) / math.log(2) + 1.0
        ntk_alpha = pow(constant(np.array(2, dtype=np.float32)), ceil(context_value)) - 1.0

        ntk_alpha = maximum(ntk_alpha, constant(np.array(1.0, dtype=np.float32)))
        base = constant(np.array(self.base, dtype=np.float32))
        base = base * pow(ntk_alpha, (self.per_head_dim / (self.per_head_dim - 2)))
        temp1 = constant(np.arange(0, self.per_head_dim, 2, dtype=np.float32) / self.per_head_dim)
        temp2 = pow(base, temp1)
        inv_freq = div(
            constant(np.array(1, dtype=np.float32)),
            temp2
        )
        # temp_length = f_max(2 * input_len, 16)
        seq = arange(constant(np.array(0, dtype=np.int32)), input_len * 2, dtype="int32")
        # with tensorrt_llm.precision("float32"):
        freqs = outer(seq.cast(trt.float32), inv_freq)
        emb = concat([freqs, freqs], dim=1)
        # emb = rearrange(emb, "n d -> 1 n 1 d")
        emb = emb.view(concat([1, input_len * 2, 1, self.per_head_dim]))
        emb = expand(emb, concat([batch_size, input_len * 2, 1, self.per_head_dim]))

        # with tensorrt_llm.precision("float32"):
        # cos, sin = emb.cos(), emb.sin()
        cos_res = cos(emb)
        sin_res = sin(emb)
        # position_embedding_cos = cos[:, :input_len]
        # position_embedding_sin = sin[:, :input_len]
        position_embedding_cos = slice(
            input=cos_res,
            starts=concat([0, 0, 0, 0]),
            sizes=concat([batch_size, input_len, 1, self.per_head_dim]),
        )
        position_embedding_sin = slice(
            input=sin_res,
            starts=concat([0, 0, 0, 0]),
            sizes=concat([batch_size, input_len, 1, self.per_head_dim]),
        )

        # self.register_network_output("my_cos", identity_op(position_embedding_cos))
        # self.register_network_output("my_sin", identity_op(position_embedding_sin))
        # expand_dims(position_embedding_cos, [batch_size, 1, 1, 1])
        rotary_pos_emb = [
            (position_embedding_cos, position_embedding_sin), 
            (position_embedding_cos, position_embedding_sin), 
        ] 
        return rotary_pos_emb



class TestFunctional(unittest.TestCase):

    per_head_dim = 128
    seq_length = 8192
    base = 10000.0
    vocab_size = 151936

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([('float32', 9886), ('float32', 1886), ('float16', 1886), ('float16', 9886)])
    def test_case(self, dtype, input_length):
        

        def test_trt(feed_dict: dict):
            # construct trt network
            builder = tensorrt_llm.Builder()
            net = builder.create_network()
            with tensorrt_llm.net_guard(net):
                input_ids = Tensor(
                    name='input_ids',
                    shape=[-1, -1],
                    dtype=trt.int32,
                    dim_range=OrderedDict([
                        ("batch_size", [[1, 1, 1]]),
                        ("seq_length", [[1, 10 * 1024, 32 * 1024]])
                    ])
                )
                # position_ids = Tensor(
                #     name='position_ids',
                #     shape=[-1, -1],
                #     dtype=trt.int32,
                #     dim_range=OrderedDict([
                #         ("batch_size", [[1, 1, 1]]),
                #         ("seq_length", [[1, 10 * 1024, 32 * 1024]])
                #     ])
                # )
                model = RotaryEmbedding(per_head_dim=self.per_head_dim, seq_length=self.seq_length)
                outputs = model.forward(input_ids=input_ids)
                # net._mark_output(outputs[0][0], 'cos', tensorrt_llm.str_dtype_to_trt(dtype))
                # net._mark_output(outputs[0][1], 'sin', tensorrt_llm.str_dtype_to_trt(dtype))
                net._mark_output(outputs[0][0], 'cos', trt.float32)
                net._mark_output(outputs[0][1], 'sin', trt.float32)

                for k, v in model.named_network_outputs():
                    # net._mark_output(v, k, tensorrt_llm.str_dtype_to_trt(dtype))
                    net._mark_output(v, k, trt.float32)
            # for build and run
            profile = Profile().add(
                "input_ids", min=(1, 1), opt=(1, 1), max=(2, 16 * 1024)
                )
            build_engine = EngineFromNetwork(
                (builder.trt_builder, net.trt_network),
                config=CreateConfig(
                    fp16=(dtype == 'float16'),
                    precision_constraints="obey",
                    profiles=[profile]
                )
            )
            with TrtRunner(build_engine) as runner:
                outputs = runner.infer(feed_dict=feed_dict) 
            return outputs
        
        def test_pytorch(input_tensor: torch.tensor):
            pt_input_len = input_tensor.shape[1]
            # upper for old
            # lower for pure pytorch for fp32 consistency(code in above used fp64 by python)
            pt_context_value = math.log(pt_input_len / self.seq_length, 2) + 1
            # pt_context_value = torch.log(torch.Tensor([input_seq_len * 1. / self.seq_length]).cuda()) / torch.log(torch.Tensor([2.]).cuda()) + 1

            pt_ntk_alpha = 2 ** math.ceil(pt_context_value) - 1
            # pt_ntk_alpha = torch.Tensor([2]).cuda() ** torch.ceil(pt_context_value) - 1

            pt_ntk_alpha = max(pt_ntk_alpha, 1.0)

            pt_ntk_alpha = pt_ntk_alpha ** (self.per_head_dim / (self.per_head_dim - 2))

            pt_base = torch.Tensor([self.base]).cuda()
            pt_base = pt_base * pt_ntk_alpha
            pt_temp1 = (torch.arange(0, self.per_head_dim, 2).float() / self.per_head_dim).cuda()
            pt_temp2 = torch.pow(pt_base, pt_temp1) # base ** temp1
            pt_inv_freq = 1.0 / pt_temp2
            pt_seq = torch.arange(0, pt_input_len * 2).int().cuda()
            pt_freqs = torch.outer(pt_seq.type_as(pt_inv_freq), pt_inv_freq)
            pt_emb = torch.cat((pt_freqs, pt_freqs), dim=-1)
            # emb = rearrange(emb, "n d -> 1 n 1 d")
            pt_emb = pt_emb.unsqueeze(0).unsqueeze(2)
            pt_cos, pt_sin = pt_emb.cos(), pt_emb.sin()
            pt_cos = pt_cos[:, :pt_input_len]
            pt_sin = pt_sin[:, :pt_input_len]
            print("pt_cos shpae/mean/sum/dtype", pt_cos.shape, pt_cos.mean(), pt_cos.sum(), pt_cos.dtype)
            print("pt_sin shpae/mean/sum/dtype", pt_sin.shape, pt_sin.mean(), pt_sin.sum(), pt_sin.dtype)
            return pt_cos, pt_sin

        

        pt_batch_size = 1
        # pt_input_len = 9886
        pt_input_len = input_length
        print("\ndtype", dtype, "input_length", input_length)
        input_tensor = torch.randint(1, self.vocab_size, [pt_batch_size, pt_input_len], dtype=torch.int32)
        # position_tensor = torch.arange(0, pt_input_len, dtype=torch.int32).unsqueeze(0).expand([pt_batch_size, pt_input_len])
        # print("position_tensor shape", position_tensor.shape)
        pt_cos, pt_sin = test_pytorch(input_tensor)
        outputs = test_trt(
            feed_dict={
                "input_ids": input_tensor.numpy(),
            }
        )

        # import pdb; pdb.set_trace()

        # np.testing.assert_allclose(ntk_alpha.cpu().numpy(), outputs['ntk_alpha'], rtol=0, atol=0)
        # np.testing.assert_allclose(base.cpu().numpy(), outputs['base'], rtol=0, atol=0)
        # np.testing.assert_allclose(temp1.cpu().numpy(), outputs['temp1'], rtol=0, atol=0)
        # np.testing.assert_allclose(temp2.cpu().numpy(), outputs['temp2'], rtol=0, atol=0)
        # np.testing.assert_allclose(seq.cpu().numpy(), outputs['seq'], rtol=1e-9, atol=1e-9)
        # np.testing.assert_allclose(inv_freq.cpu().numpy(), outputs['inv_freq'], rtol=1e-9, atol=1e-9)
        # np.testing.assert_allclose(pt_freqs.cpu().numpy(), outputs['freqs'], rtol=1e-9, atol=1e-9)
        print("cos shpae/mean/sum/dtype", outputs["cos"].shape, outputs["cos"].mean(), outputs["cos"].sum(), outputs["cos"].dtype)
        print("sin shpae/mean/sum/dtype", outputs["sin"].shape, outputs["sin"].mean(), outputs["sin"].sum(), outputs["sin"].dtype)
        np.testing.assert_allclose(pt_cos.cpu().numpy(), outputs['cos'], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(pt_sin.cpu().numpy(), outputs['sin'], rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    unittest.main()