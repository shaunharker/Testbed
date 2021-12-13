import math
import torch
from torch.nn import Module, Linear, Dropout, LayerNorm
from torch.cuda.amp import autocast
from .nn import Sequential, Embedding, MLP, LanguageModel, ResidualDropoutLayerNorm


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
    def __init__(self, d_model, d_k, d_v, n_heads, p_dropout):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.p_dropout = p_dropout

        self.query_proj = Linear(d_model, d_k*n_heads)
        self.key_proj = Linear(d_model, d_k*n_heads)
        self.value_proj = Linear(d_model, d_v*n_heads)
        self.mask = Mask(mode="causal")
        self.dropout = Dropout(p_dropout)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.linear = Linear(d_v*n_heads, d_model, bias=False)

    def forward(self, x):
        (n_ctx, d_model) = x.shape[-2:]
        assert d_model == self.d_model, f"{d_model} != {self.d_model}"
        split_heads = lambda x: x.view(x.shape[:-1]+(self.n_heads,-1)).transpose(-2,-3).contiguous()
        merge_heads = lambda x: x.transpose(-2,-3).contiguous().view(x.shape[:-3]+(n_ctx,self.d_v*self.n_heads))
        (Q, K, V) = map(split_heads,(self.query_proj(x),self.key_proj(x),self.value_proj(x)))
        QKT = torch.matmul(Q/math.sqrt(self.d_k),K.transpose(-1,-2))
        return self.linear(merge_heads(self.dropout(self.softmax(self.mask(QKT)))@V))


class FastformerAttn(Module):
    def __init__(self, d_model, d_q, d_k, d_v, n_heads):
        super().__init__()
        self.d_model = d_model
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.split_heads = lambda x: x.view(x.shape[:-1]+(self.n_heads,-1)).transpose(-2,-3).contiguous()
        self.query_transformation = Sequential(
            Linear(d_model, d_q*n_heads),
            Lambda(self.split_heads))
        self.key_transformation = Sequential(
            Linear(d_model, d_k*n_heads),
            Lambda(self.split_heads))
        self.value_transformation = Sequential(
            Linear(d_model, d_v*n_heads),
            Lambda(self.split_heads))
        self.softmax = torch.nn.Softmax(dim=-1)
        self.alpha = Sequential(Linear(d_q, 1), torch.nn.Softmax(dim=-2))
        self.beta = Sequential(Linear(d_k, 1), torch.nn.Softmax(dim=-2))
        self.merge_heads = lambda x: x.transpose(-2,-3).contiguous().view(x.shape[:-3]+(n_ctx,self.d_v*self.n_heads))
    def forward(self, x):
        (n_ctx, d_model) = x.shape[-2:]
        assert d_model == self.d_model, f"{d_model} != {self.d_model}"

        (Q, K, V) = map(split_heads,(self.query_proj(x),self.key_proj(x),self.value_proj(x)))
        # Q =
        QKT = torch.matmul(Q/math.sqrt(self.d_k),K.transpose(-1,-2))
        return self.linear(merge_heads(self.dropout(self.softmax(self.mask(QKT)))@V))


class FastformerLayer(Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_hidden):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_hidden = d_hidden


    def forward(self, x):
        return self.mlp(self.attn(x))


class Fastformer(Module):
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
        self.language_model = (
            LanguageModel(
                n_vocab_out=n_vocab_out,
                mode="last",
                module=(None)))  # TODO

    def forward(self, x):
        with autocast(enabled=self.autocast_enabled):
            return self.language_model(x)

    @torch.no_grad()
    def inference(self, x):
        with autocast(enabled=self.autocast_enabled):
            return self.language_model.inference(x)

    def clone(self):
        return copy.deepcopy(self)
