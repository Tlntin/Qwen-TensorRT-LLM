import unittest

from parameterized import parameterized

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_np
from tensorrt_llm.models import GPTLMHeadModel, OPTLMHeadModel


class TestModelDtype(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([(GPTLMHeadModel, 'float32'),
                           (GPTLMHeadModel, 'bfloat16'),
                           (GPTLMHeadModel, 'float16'),
                           (OPTLMHeadModel, 'float16')])
    def test_model_dtype(self, model_cls, dtype):
        ''' Every parameter in the model should have the same dtype as the model initialized to
        '''
        tiny_model = model_cls(num_layers=6,
                               num_heads=4,
                               hidden_size=128,
                               vocab_size=128,
                               hidden_act='relu',
                               max_position_embeddings=128,
                               dtype=dtype)
        for p in tiny_model.parameter():
            self.assertEqual(p._value.dtype, str_dtype_to_np(dtype))


if __name__ == '__main__':
    unittest.main()
