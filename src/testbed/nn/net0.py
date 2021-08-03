import math
import torch
from torch.nn import Module, Embedding, Linear, CrossEntropyLoss, Softmax, Sigmoid, ReLU, GELU
from torch.cuda.amp import autocast
from .mlp import MLP
from .sequential import Sequential
from .lambdamodule import Lambda


class Net0(Module):
    def __init__(self,
                 n_vocab_in=256,
                 n_vocab_out=256,
                 n_ctx=128,
                 d_model=32,
                 d_ff=8192,
                 nonlinearity='sigmoid',
                 use_amp=False):
        super().__init__()
        self.n_vocab_in = n_vocab_in
        self.n_vocab_out = n_vocab_out
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.d_ff = d_ff
        self.nonlinearity = nonlinearity
        self.use_amp = use_amp

        self.module = (
            Sequential(
                Embedding(n_vocab_in, d_model),
                Lambda(lambda x: x.view(-1,n_ctx*d_model)),
                Linear(n_ctx*d_model, d_ff),
                {"sigmoid": Sigmoid(), "ReLU": ReLU(), "GELU": GELU()}[nonlinearity],
                Linear(d_ff, n_vocab_out)))
        self.criterion = CrossEntropyLoss(reduction='none')
        self.softmax = Softmax(dim=-1)

    def forward(self, xy):
        with autocast(enabled=self.use_amp):
            assert xy.shape[-1] == self.n_ctx + 1
            (x, y) = (xy[...,:-1].contiguous(), xy[...,-1].contiguous())
            x = self.module(x)
            return self.criterion(x.reshape(-1,self.n_vocab_out),
                                  y.reshape(-1)).view(x.shape[:-1])/math.log(self.n_vocab_out)

    def probs(self, x):
        with torch.no_grad():
            return self.softmax(self.module(x))

    def name(self):
        return f"Net0({self.n_vocab_in},{self.n_vocab_out},{self.n_ctx},{self.d_model},{self.d_ff},{self.nonlinearity})"
