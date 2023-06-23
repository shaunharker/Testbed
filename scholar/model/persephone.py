import math
import torch
import copy
from torch.nn import Module, Linear, LayerNorm, Embedding
from torch.cuda.amp import autocast
from .nn import Sequential, MLP, LanguageModel, ResidualLayerNorm


class Persephone(Module):
    def __init__(self):
        super().__init__()
        self.language_model = (
            LanguageModel(
                n_vocab_out=n_vocab_out,
                mode=mode,
                module=
                    Sequential(
                        Embedding(n_vocab_in, d_model),
                        PositionalEncoding(n_ctx, d_model),
                        Sequential(layer for layer in self.layers),
                        Linear(d_model, n_vocab_out))))

    def forward(self, x):
        with autocast(enabled=self.autocast_enabled):
            return self.language_model(x)

    @torch.no_grad()
    def inference(self, x):
        with autocast(enabled=self.autocast_enabled):
            return self.language_model.inference(x)

    def clone(self):
        return copy.deepcopy(self)
