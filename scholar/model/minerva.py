import math
import torch
from torch.nn import Module, Linear, Dropout, LayerNorm
from torch.cuda.amp import autocast
from .nn import Sequential, Embedding, MLP, LanguageModel, ResidualDropoutLayerNorm


class MinervaConfig:
    def __init__(
        self,
        n_vocab,
        n_ctx,
        d_embd,
        d_model,
        n_layers,
        n_iterates,
        d_hidden,
        nonlinearity,
        p_dropout):
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.d_embd = d_embd
        self.d_model = d_model

        self.n_layers = n_layers
        self.n_iterates = n_iterates
        self.d_hidden = d_hidden
        self.nonlinearity = nonlinearity
        self.p_dropout = p_dropout

class MinervaNLM(Module):
    def __init__(self,
                 config):
        super().__init__()
        self.config = config

        self.embedding = Embedding(
            n_classes=config.n_vocab,
            d_model=config.d_embd)

        self.read = MLP(
            d_in=config.n_ctx*config.d_embd,
            d_hidden=config.d_hidden,
            nonlinearity=config.nonlinearity,
            d_out=config.d_model)

        self.thunk = MLP(
            d_model=config.d_model,
            d_hidden=config.d_hidden,
            nonlinearity=config.nonlinearity,
            p_dropout=config.p_dropout)

        self.think = ResidualDropoutLayerNorm(
            layer=self.thunk,
            d_model=config.d_model,
            p_dropout=config.p_dropout)

        self.write = MLP(
            d_in=config.d_model,
            d_hidden=config.d_hidden,
            nonlinearity=config.nonlinearity,
            d_out=config.n_vocab_out)

        self.split_example = SplitExample("last")
        self.crossentropyloss = CrossEntropyLoss(n_vocab_out)
        self.softmax = Softmax()

    def F(self, x0):
        x = self.embedding(x0)
        x = x.view(-1, self.config.n_ctx*self.config.d_embd)
        x = self.read(x)
        for _ in range(self.config.n_iterations):
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
