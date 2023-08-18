from typing import Optional, Tuple

from ..functional import avg_pool2d
from ..module import Module


class AvgPool2d(Module):

    def __init__(self,
                 kernel_size: Tuple[int],
                 stride: Optional[Tuple[int]] = None,
                 padding: Optional[Tuple[int]] = (0, 0),
                 ceil_mode: bool = False,
                 count_include_pad: bool = True) -> None:
        super().__init__()
        self.kernel_szie = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        return avg_pool2d(input, self.kernel_szie, self.stride, self.padding,
                          self.ceil_mode, self.count_include_pad)
