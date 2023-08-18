from ..functional import ACT2FN
from ..module import Module
from .linear import ColumnLinear, RowLinear


class MLP(Module):

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 hidden_act,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1):
        super().__init__()
        if hidden_act not in ACT2FN:
            raise ValueError(
                'unsupported activation function: {}'.format(hidden_act))
        self.fc = ColumnLinear(hidden_size,
                               ffn_hidden_size,
                               bias=bias,
                               dtype=dtype,
                               tp_group=tp_group,
                               tp_size=tp_size,
                               gather_output=False)
        self.proj = RowLinear(ffn_hidden_size,
                              hidden_size,
                              bias=bias,
                              dtype=dtype,
                              tp_group=tp_group,
                              tp_size=tp_size)
        self.hidden_act = hidden_act
        self.dtype = dtype

    def forward(self, hidden_states):
        inter = self.fc(hidden_states)
        inter = ACT2FN[self.hidden_act](inter)
        output = self.proj(inter)
        return output


class GatedMLP(MLP):

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 hidden_act,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1):
        super().__init__(hidden_size,
                         ffn_hidden_size,
                         hidden_act,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size)
        self.gate = ColumnLinear(hidden_size,
                                 ffn_hidden_size,
                                 bias=bias,
                                 dtype=dtype,
                                 tp_group=tp_group,
                                 tp_size=tp_size,
                                 gather_output=False)

    def forward(self, hidden_states):
        inter = self.fc(hidden_states)
        inter = ACT2FN[self.hidden_act](inter)
        gate = self.gate(hidden_states)
        output = self.proj(inter * gate)
        return output
