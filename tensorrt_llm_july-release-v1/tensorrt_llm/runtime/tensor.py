from typing import Sequence

import torch


class RaggedTensor:

    def __init__(self, **kwargs):
        """Do not call this directly, call the from_row_lengths instead
        """

    def _init(self, data: torch.Tensor, row_lengths: torch.Tensor,
              ragged_dim: int, is_data_packed: bool):
        self._data = data
        self._row_lengths = row_lengths
        self._ragged_dim = ragged_dim
        self._is_data_packed = is_data_packed
        if self.is_data_packed:
            assert self.data.size(
                0) == 1, "Must prefix a 1 in the dims for packed format"
        return self

    @property
    def data(self):
        return self._data

    @property
    def row_lengths(self):
        return self._row_lengths

    @property
    def ragged_dim(self):
        return self._ragged_dim

    @property
    def is_data_packed(self):
        return self._is_data_packed

    @staticmethod
    def from_row_lengths(data: torch.Tensor, row_lengths: torch.Tensor,
                         ragged_dim: int, is_data_packed):
        return RaggedTensor()._init(data, row_lengths, ragged_dim,
                                    is_data_packed)

    @staticmethod
    def from_tensors(tensors: Sequence[torch.Tensor]):
        tensors = [torch.flatten(t) for t in tensors]
        data = torch.unsqueeze(torch.concat(tensors), 0)
        row_lengths = [t.size(0) for t in tensors]
        row_lengths = torch.tensor(row_lengths,
                                   dtype=torch.int32,
                                   device=data.device)
        return RaggedTensor.from_row_lengths(data,
                                             row_lengths,
                                             1,
                                             is_data_packed=True)
