from ..functional import softplus, tanh
from ..module import Module


class Mish(Module):

    def forward(self, input):
        return input * tanh(softplus(input, beta=1.0, threshold=20.0))
