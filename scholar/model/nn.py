import math
import dill
from types import GeneratorType
import copy
import torch
from torch.cuda.amp import autocast
from torch.nn import Module, ModuleList, Sigmoid, ReLU, GELU, LayerNorm
from torch.nn import Embedding as TorchEmbedding
from torch.nn import Linear as TorchAffine


class SplitExample(Module):
    def __init__(self, mode="last"):
        super().__init__()
        self.mode = mode

    def forward(self, xy):
        if self.mode == "last":
            return (xy[...,:-1].contiguous(), xy[...,-1].contiguous())
        elif self.mode == "shift":
            return (xy[...,:-1].contiguous(), xy[...,1:].contiguous())


class Embedding(TorchEmbedding):
    def __init__(self, n_classes, d_model):
        super().__init__(n_classes, d_model)
        self.n_classes = n_classes
        self.d_model = d_model


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        layers = sum([list(layer) if type(layer)==GeneratorType else [layer] for layer in layers],[])
        self.layers = ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Lambda(Module):
    def __init__(self, F):
        super().__init__()
        self.F = F

    def forward(self, x):
        return self.F(x)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['F'] = dill.dumps(self.F)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.F = dill.loads(self.F)


class Affine(TorchAffine):
    def __init__(self, d_in, d_out, bias=True):
        super().__init__(d_in, d_out, bias)
        self.d_in = d_in
        self.d_out = d_out


class Nonlinearity(Module):
    def __init__(self, nonlinearity):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.f = {"sigmoid": Sigmoid(), "ReLU": ReLU(), "GELU": GELU()}[nonlinearity]

    def forward(self, x):
        return self.f(x)


class CrossEntropyLoss(Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.crossentropyloss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, y):
        return self.crossentropyloss(x.reshape(-1,self.n_classes), y.reshape(-1)).view(x.shape[:-1])/math.log(self.n_classes)


class Softmax(Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(x)


class ResidualLayerNorm(Module):
    def __init__(self, layer, d_model):
        super().__init__()
        self.d_model = d_model
        self.layer = layer
        self.layernorm = LayerNorm(d_model)

    def forward(self, x):
        assert x.shape[-1] == self.d_model, f"{x.shape[-1]} != {self.d_model}"
        return self.layernorm(x+self.layer(x))


class MLP(Module):
    def __init__(self, d_in, d_hidden, nonlinearity, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden if type(d_hidden) == list else [d_hidden]
        self.d_out = d_out
        self.nonlinearity = nonlinearity
        self.module = Sequential(
            Affine(d_in=d_in, d_out=self.d_hidden[0]),
            Sequential(
                Sequential(
                    Nonlinearity(nonlinearity),
                    Affine(d_in=a, d_out=b))
                for (a,b) in zip(self.d_hidden[:-1], self.d_hidden[1:])),
            Nonlinearity(nonlinearity),
            Affine(d_in=self.d_hidden[-1], d_out=d_out))

    def forward(self, x):
        return self.module(x)


class LanguageModel(Module):
    def __init__(self, n_vocab_out, mode, module):
        super().__init__()
        self.n_vocab_out = n_vocab_out
        self.mode = mode
        self.module = module
        self.split_example = SplitExample(mode)
        self.crossentropyloss = CrossEntropyLoss(n_vocab_out)
        self.softmax = Softmax()

    def forward(self, xy):
        (x, y) = self.split_example(xy)
        x = self.module(x)
        return self.crossentropyloss(x, y)

    @torch.no_grad()
    def inference(self, x):
        return self.softmax(self.module(x))
