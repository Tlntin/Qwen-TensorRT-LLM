from typing import Sequence, Union

import numpy as np
import tensorrt as trt

from ._utils import str_dtype_to_trt, torch_to_numpy, trt_dtype_to_torch
from .functional import Tensor, constant
from .logger import logger


class Parameter(object):
    _DEFAULT_DTYPE = trt.DataType.FLOAT

    def __init__(self,
                 value: Union[np.ndarray] = None,
                 shape: Sequence[int] = None,
                 dtype: Union[str, trt.DataType] = None):
        if dtype is None:
            logger.warning(
                f'Parameter dtype is None, using default dtype: {self._DEFAULT_DTYPE}, it is recommended to always specify dtype explicitly'
            )
        dtype = self._DEFAULT_DTYPE if dtype is None else dtype
        if isinstance(dtype, str):
            dtype = str_dtype_to_trt(dtype)
        if value is None:
            import torch
            assert isinstance(shape, (list, tuple))
            if len(shape) == 2:
                # Xavier initialization see https://paperswithcode.com/method/xavier-initialization
                v_range = np.sqrt(6) / np.sqrt(shape[0] + shape[1])
            else:
                v_range = 0.1

            # value ~ U[-1, 1]
            value = torch.rand(
                (shape), dtype=trt_dtype_to_torch(dtype), device='cuda') * 2 - 1
            # value ~ U[-v_range, v_range]
            value = torch_to_numpy((value * v_range).cpu())

        self._value = value

    @property
    def value(self) -> Tensor:
        if isinstance(self._value, np.ndarray):
            self._value = constant(self._value)

        return self._value

    @value.setter
    def value(self, v: np.ndarray):
        assert isinstance(v, np.ndarray)
        assert v.shape == self._value.shape, \
            ('The value updated is not the same shape as the original. ', \
            f'Updated: {v.shape}, original: {self._value.shape}')
        self._value = v

    def _get_weights(self) -> trt.Weights:
        return self._value.producer.weights if isinstance(self._value,
                                                          Tensor) else None
