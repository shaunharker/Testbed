import math
import dill
from types import GeneratorType
import copy
import torch
from torch.cuda.amp import autocast
from torch.nn import Module, ModuleList, Sigmoid, ReLU, GELU, LayerNorm
from torch.nn import Embedding as TorchEmbedding
from torch.nn import Linear as TorchAffine
from torch.nn import Dropout


class SplitExample(Module):
    def __init__(self, mode="last"):
        super().__init__()
        self.mode = mode

    def forward(self, xy):
        if self.mode == "last":
            return (xy[...,:-1].contiguous(), xy[...,-1].contiguous())
        elif self.mode == "shift":
            n = xy.shape[-1]
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


class MLPLM(Module):
    def __init__(self, n_ctx, n_vocab_in, d_model, d_hidden, nonlinearity, n_vocab_out, autocast_enabled=None):
        super().__init__()
        self.n_ctx = n_ctx
        self.n_vocab_in = n_vocab_in
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.nonlinearity = nonlinearity
        self.n_vocab_out = n_vocab_out
        self.autocast_enabled = autocast_enabled or False
        self.language_model = (
            LanguageModel(
                n_vocab_out=n_vocab_out,
                mode="last",
                module=(
                    Sequential(
                        Embedding(n_classes=n_vocab_in, d_model=d_model),
                        Lambda(lambda x: x.view(-1,n_ctx*d_model)),
                        MLP(d_in=n_ctx*d_model,
                            d_hidden=d_hidden,
                            nonlinearity=nonlinearity,
                            d_out=n_vocab_out),
                        Lambda(lambda x: x.view(-1, 1, n_vocab_out))))))

    def forward(self, x):
        with autocast(enabled=self.autocast_enabled):
            return self.language_model(x)

    @torch.no_grad()
    def inference(self, x):
        with autocast(enabled=self.autocast_enabled):
            return self.language_model.inference(x)

    def clone(self):
        return copy.deepcopy(self)


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


class MyLM(Module):
    def __init__(self, n_ctx, n_vocab_in, d_model, n_layers, d_hidden, nonlinearity, p_dropout, n_vocab_out, autocast_enabled=None):
        super().__init__()
        self.n_ctx = n_ctx
        self.n_vocab_in = n_vocab_in
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_hidden = d_hidden
        self.nonlinearity = nonlinearity
        self.p_dropout = p_dropout
        self.n_vocab_out = n_vocab_out
        self.autocast_enabled = autocast_enabled or False
        self.language_model = (
            LanguageModel(
                n_vocab_out=n_vocab_out,
                mode="last",
                module=(
                    Sequential(
                        Embedding(n_classes=n_vocab_in, d_model=d_model),
                        Lambda(lambda x: x.view(-1,n_ctx*d_model)),
                        *[ResidualDropoutLayerNorm(
                            layer=MyLayer(
                                d_model=n_ctx*d_model,
                                d_hidden=d_hidden,
                                nonlinearity=nonlinearity,
                                p_dropout=p_dropout),
                            d_model=n_ctx*d_model,
                            p_dropout=p_dropout) for _ in range(n_layers)],
                        MLP(d_in=n_ctx*d_model,
                            d_hidden=d_hidden,
                            nonlinearity=nonlinearity,
                            d_out=n_vocab_out),
                        Lambda(lambda x: x.view(-1, 1, n_vocab_out))))))

    def forward(self, x):
        with autocast(enabled=self.autocast_enabled):
            return self.language_model(x)

    @torch.no_grad()
    def inference(self, x):
        with autocast(enabled=self.autocast_enabled):
            return self.language_model.inference(x)

    def clone(self):
        return copy.deepcopy(self)


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
        elif self.mode == "half_causal":
            return x+(1-1/torch.cat([torch.cat([torch.ones((n//2,n//2),device=device), torch.zeros((n//2,n//2),device=device)], dim=1), torch.tril(torch.ones((n,n),device=device))[n//2:,:]], dim=0))


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

        self.embedding = (
            Embedding(
                n_classes=config.n_vocab,
                d_model=config.d_embd))

        self.read = (
            MLP(d_in=n_ctx*d_embd,
                d_hidden=d_hidden,
                nonlinearity=nonlinearity,
                d_out=d_model))

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
