import torch
from torch import nn
from torch.nn import Module, Sequential, Linear, ReLU


class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def _fold(self, x, target_dim):
        in_dim = x.size(-1)
        folded = torch.zeros_like(x[..., :target_dim])
        for i in range(in_dim):
            folded[..., i % target_dim] += x[..., i]
        return folded

    def _match_dimensions(self, x, fx):
        in_dim, out_dim = x.size(-1), fx.size(-1)

        if in_dim == out_dim:
            return x, fx
        elif in_dim < out_dim:
            x_expanded = torch.cat([x] + [torch.zeros_like(x) for _ in range(out_dim - in_dim)], dim=-1)
            return x_expanded, fx
        else:
            x_folded = self._fold(x, out_dim)
            return x_folded, fx

    def forward(self, x):
        fx = self.module(x)
        x, fx = self._match_dimensions(x, fx)
        return x + fx
    

class MLPWithResidualLayers(Module):
    def __init__(self, shape, nonlinearity = ReLU()):
        super().__init__()
        self.shape = shape
        self.nonlinearity = nonlinearity
        
        layers = []
        for i in range(len(shape) - 2):
            wrapped_layer = Residual(Sequential(Linear(shape[i], shape[i + 1]), nonlinearity))
            layers.append(wrapped_layer)

        layers.append()
        self.module = Sequential(*layers)

    def forward(self, x):
        return self.module(x)

