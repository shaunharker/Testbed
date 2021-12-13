import math
import torch
from torch.nn import Module, Linear, Dropout, LayerNorm
from torch.cuda.amp import autocast
from .nn import Sequential, Embedding, MLP, LanguageModel, ResidualDropoutLayerNorm





class RDLNMLP(Module):
    def __init__(self, d_model, d_hidden, nonlinearity, p_dropout):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.nonlinearity = nonlinearity
        self.p_dropout = p_dropout
        self.rdln = (
            ResidualDropoutLayerNorm(
                Sequential(
                    Affine(d_in=d_model, d_out=d_hidden),
                    Nonlinearity(nonlinearity),
                    Affine(d_in=d_hidden, d_out=d_model)),
                d_model = d_model,
                p_dropout = p_dropout))

    def forward(self, x):
        return self.rdln(x)


class MyLayer(Module):
    def __init__(self, d_model, d_hidden, nonlinearity, p_dropout):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.nonlinearity = nonlinearity
        self.p_dropout = p_dropout
        self.A = RDLNMLP(d_model, d_hidden, nonlinearity, p_dropout)
        self.B = RDLNMLP(d_model, d_hidden, nonlinearity, p_dropout)
        self.C = RDLNMLP(d_model, d_hidden, nonlinearity, p_dropout)

    def forward(self, x):
        return self.A(x)*self.B(x)+self.C(x)


class ABPCNLM(Module):
    def __init__(self, n_ctx, n_vocab_in, d_model, n_layers, d_hidden, nonlinearity, p_dropout, n_vocab_out):
        super().__init__()
        self.n_ctx = n_ctx
        self.n_vocab_in = n_vocab_in
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_hidden = d_hidden
        self.nonlinearity = nonlinearity
        self.p_dropout = p_dropout
        self.n_vocab_out = n_vocab_out

        self.embedding = Embedding(n_classes=n_vocab_in, d_model=d_model)

        self.read = (
            MLP(d_in=n_ctx*d_model,
                d_hidden=d_hidden,
                nonlinearity=nonlinearity,
                d_out=n_ctx*d_model))

        self.think = (
            ResidualDropoutLayerNorm(
                layer=MyLayer(
                    d_model=n_ctx*d_model,
                    d_hidden=d_hidden,
                    nonlinearity=nonlinearity,
                    p_dropout=p_dropout),
                d_model=n_ctx*d_model,
                p_dropout=p_dropout))

        self.write = (
            MLP(d_in=n_ctx*d_model,
                d_hidden=d_hidden,
                nonlinearity=nonlinearity,
                d_out=self.n_vocab_out))

        self.split_example = SplitExample("last")
        self.crossentropyloss = CrossEntropyLoss(n_vocab_out)
        self.softmax = Softmax()

    def F(self, x0):
        x = self.embedding(x0)
        x = x.view(-1, self.n_ctx*self.d_model)
        x = self.read(x)
        for _ in range(self.n_layers):
            x = self.think(x)
        x = self.write(x)
        return x

    def forward(self, xy):
        (x, y) = self.split_example(xy)
        return self.crossentropyloss(self.F(x), y)

    @torch.no_grad()
    def inference(self, x):
        return self.softmax(self.F(x))

    def clone(self):
        return copy.deepcopy(self)
