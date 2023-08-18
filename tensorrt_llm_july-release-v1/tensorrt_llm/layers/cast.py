from ..functional import cast
from ..module import Module


class Cast(Module):

    def __init__(self, output_dtype: str = 'float32') -> None:
        super().__init__()
        assert output_dtype in ('float32', 'float16', 'bfloat16', 'bool',
                                'int32', 'int8'), TypeError(
                                    "%s is not supported" % output_dtype)
        self.output_dtype = output_dtype

    def forward(self, x):
        return cast(x, self.output_dtype)
