from torch.nn import Module, Dropout, LayerNorm
from torch.cuda.amp import autocast


class ResidualDropoutLayerNorm(Module):
    def __init__(self, layer, d_model, p_dropout):
        super().__init__()
        self.d_model = d_model
        self.p_dropout = p_dropout

        self.layer = layer
        self.dropout = Dropout(p_dropout)
        self.layernorm = LayerNorm(d_model)

    def forward(self, x):
        assert x.shape[-1] == self.d_model, f"{x.shape[-1]} != {self.d_model}"
        return self.layernorm(x+self.dropout(self.layer(x)))
