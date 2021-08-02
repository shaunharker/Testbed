import torch
from torch.nn import Module, Linear, Sigmoid, ReLU, GELU
from torch.cuda.amp import autocast
from .sequential import Sequential


class MLP(Module):
    def __init__(self, d_model, d_ff, nonlinearity="GELU"):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.nonlinearity = nonlinearity
        self.mlp = (
            Sequential(
                Linear(d_model, d_ff),
                {"sigmoid": Sigmoid(), "ReLU": ReLU(), "GELU": GELU()}[nonlinearity],
                Linear(d_ff, d_model, bias=False)))

    @autocast()
    def forward(self, x):
        return self.mlp(x)
