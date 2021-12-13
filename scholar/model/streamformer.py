import math
import torch
from torch.nn import Module, Linear, Dropout, LayerNorm
from torch.cuda.amp import autocast
from .nn import Sequential, Embedding, MLP, LanguageModel, Lambda


class ResidualLayerNorm(Module):
    def __init__(self, layer, d_model):
        super().__init__()
        self.d_model = d_model

        self.layer = layer
        self.layernorm = LayerNorm(d_model)

    def forward(self, x):
        assert x.shape[-1] == self.d_model, f"{x.shape[-1]} != {self.d_model}"
        return self.layernorm(x+self.layer(x))


class Mask(Module):
    def __init__(self, mode="none"):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        n, device = x.shape[-1], x.device
        if self.mode == "none":
            return x
        elif self.mode == "causal":
            return x+(1-1/torch.tril(torch.ones((n,n),device=device)))


class Attn(Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.query_proj = Linear(d_model, d_k*n_heads)
        self.key_proj = Linear(d_model, d_k*n_heads)
        self.value_proj = Linear(d_model, d_v*n_heads)
        self.mask = Mask(mode="none")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.linear = Linear(d_v*n_heads, d_model, bias=False)

    def forward(self, x):
        (n_ctx, d_model) = x.shape[-2:]
        assert d_model == self.d_model, f"{d_model} != {self.d_model}"
        split_heads = lambda x: x.view(x.shape[:-1]+(self.n_heads,-1)).transpose(-2,-3).contiguous()
        merge_heads = lambda x: x.transpose(-2,-3).contiguous().view(x.shape[:-3]+(n_ctx,self.d_v*self.n_heads))
        (Q, K, V) = map(split_heads,(self.query_proj(x),self.key_proj(x),self.value_proj(x)))
        QKT = torch.matmul(Q/math.sqrt(self.d_k),K.transpose(-1,-2))
        return self.linear(merge_heads(self.softmax(self.mask(QKT))@V))


class StreamformerLayer(Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_hidden):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_hidden = d_hidden

        self.attn = ResidualLayerNorm(Attn(d_model, d_k, d_v, n_heads), d_model)
        self.mlp = ResidualLayerNorm(MLP(d_model, d_hidden, 'GELU', d_model), d_model)

    def forward(self, x):
        return self.mlp(self.attn(x))


class PositionalEncoding(Module):
    def __init__(self, n_ctx, d_model):
        super().__init__()
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.weight = torch.nn.Parameter(0.02*torch.randn(n_ctx, d_model))

    def forward(self, x):
        n_ctx = x.shape[-2]
        assert n_ctx <= self.n_ctx
        return x + self.weight[:n_ctx]


class StreamformerLM(Module):
    def __init__(self,
                 n_vocab_in,
                 n_vocab_out,
                 n_ctx,
                 d_model,
                 d_k,
                 d_v,
                 n_heads,
                 d_hidden,
                 n_layers,
                 autocast_enabled=None):
        super().__init__()
        self.n_vocab_in = n_vocab_in
        self.n_vocab_out = n_vocab_out
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.autocast_enabled = autocast_enabled or False
        self.language_model = (
            LanguageModel(
                n_vocab_out=n_vocab_out,
                mode="shift",
                module=
                    Sequential(
                        Embedding(n_vocab_in, d_model),
                        PositionalEncoding(n_ctx, d_model),
                        Sequential(
                            StreamformerLayer(d_model, d_k, d_v, n_heads, d_hidden) for _ in range(n_layers)),
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
