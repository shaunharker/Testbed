import torch
from math import sqrt, log
from torch.nn import Module, Linear, Sequential, Embedding, LayerNorm, Sigmoid, ReLU, GELU


class Nonlinearity(Module):
    def __init__(self, nonlinearity):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.f = {"sigmoid": Sigmoid(), "ReLU": ReLU(), "GELU": GELU()}[nonlinearity]

    def forward(self, x):
        return self.f(x)


class MLP(Module):
    def __init__(self, config):
        super().__init__()
        m = config["d_model"]
        n = config["d_hidden"]
        self.model = Sequential(
            Linear(m, n, bias=True),
            Nonlinearity(config["nonlinearity"]),
            Linear(n, m, bias=True))

    def forward(self, x):
        return self.model(x)


class Mask(Module):
    def __init__(self, mask="none"):
        super().__init__()
        self.mask = mask

    def forward(self, x):
        n, device = x.shape[-1], x.device
        if self.mask == "none":
            return x
        elif self.mask == "causal":
            return x+(1-1/torch.tril(torch.ones((n,n),device=device)))


class Attn(Module):
    def __init__(self, config):
        super().__init__()
        d_model = self.d_model = config["d_model"]
        d_k = self.d_k = config["d_k"]
        d_v = self.d_v = config["d_v"]
        n_heads = self.n_heads = config["n_heads"]
        self.query_proj = Linear(d_model, d_k*n_heads)
        self.key_proj = Linear(d_model, d_k*n_heads)
        self.value_proj = Linear(d_model, d_v*n_heads)
        self.mask = Mask(mask=config["mask"])
        self.softmax = torch.nn.Softmax(dim=-1)
        self.linear = Linear(d_v*n_heads, d_model, bias=False)

    def forward(self, x):
        n_ctx = x.shape[-2]
        split_heads = (lambda x: x.view(x.shape[:-1] +
            (self.n_heads, -1)).transpose(-2,-3).contiguous())
        merge_heads = (lambda x: x.transpose(-2,-3).contiguous()
            .view(x.shape[:-3] + (n_ctx, self.d_v*self.n_heads)))
        (Q, K, V) = map(split_heads,(self.query_proj(x),
            self.key_proj(x), self.value_proj(x)))
        QKT = torch.matmul(Q/sqrt(self.d_k), K.transpose(-1,-2))
        U = self.softmax(self.mask(QKT))
        return self.linear(merge_heads(U@V))


class ResidualLayerNorm(Module):
    def __init__(self, layer, d_model):
        super().__init__()
        self.d_model = d_model
        self.layer = layer
        self.layernorm = LayerNorm(d_model)

    def forward(self, x):
        return self.layernorm(x+self.layer(x))


class TransformerLayer(Module):
    def __init__(self, config):
        super().__init__()
        d_model = config["d_model"]
        self.model = Sequential(
            ResidualLayerNorm(Attn(config), d_model),
            ResidualLayerNorm(MLP(config), d_model))

    def forward(self, x):
        return self.model(x)


class PositionalEncoding(Module):
    def __init__(self, config):
        super().__init__()
        n_ctx = config["n_ctx"]
        d_model = config["d_model"]
        init_weights = 0.02*torch.randn(n_ctx, d_model)
        self.weight = torch.nn.Parameter(init_weights)

    def forward(self, x):
        n_ctx = x.shape[-2]
        return x + self.weight[:n_ctx]


class View(Module):
    def __init__(self, *suffix):
        super().__init__()
        self.suffix = suffix

    def forward(self, x):
        return x.view(*x.shape[:-1], *self.suffix)

class ChessLanguageModel(Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = {
            "n_classes": 256,
            "n_ctx": 4096,
            "n_layers": 3,
            "plan": [0,1,2],
            "d_model": 1024,
            "d_k": 32,
            "d_v": 32,
            "n_heads": 32,
            "d_hidden": 1024,
            "nonlinearity": "GELU",
            "mask": "causal",
            "device": "cuda"}
        self.config.update(config or dict())
        n_ctx = self.config["n_ctx"]
        n_layers = self.config["n_layers"]
        d_model = self.config["d_model"]
        plan = self.config["plan"]
        device = self.config["device"]
        make_layer = lambda: TransformerLayer(self.config)
        self.layers = [make_layer() for _ in range(n_layers)]
        self.model = Sequential(
            Embedding(256, d_model),
            PositionalEncoding(self.config),
            Sequential(*[self.layers[i] for i in plan]))
        self.seq_head = Linear(d_model, 256)
        self.visual_head = Sequential(Linear(d_model, 64*13), View(64, 13))
        self.action_head = Sequential(Linear(d_model, 256*2), View(256, 2))
        self.crossentropyloss = torch.nn.CrossEntropyLoss(reduction='none')
        self.softmax = torch.nn.Softmax(dim=-1)
        self.to(device)

    def forward(self, seq_input, seq_target, visual_target, action_target):
        model_output = self.model(seq_input)
        seq_output = self.seq_head(model_output)
        visual_output = self.visual_head(model_output)
        action_output = self.action_head(model_output)
        print('A', seq_output.shape, visual_output.shape, action_output.shape)
        # Per seq index, we get a 256 prediction
        seq_loss = self.crossentropyloss(
            seq_output.view(-1, 256),
            seq_target.view(-1)
        ).view(seq_output.shape[:-1])/log(256)
        # Per seq index, we get a 64x13 .pnkqbrPNKQBR
        visual_loss = self.crossentropyloss(
            visual_output.view(-1, 13),
            visual_target.view(-1)
        ).view(visual_output.shape[:-1])/log(13)
        # Per seq index, we get a 256x2 matrix for legal seq outputs
        # where each row can be softmaxed to give probabilities
        action_loss = self.crossentropyloss(
            action_output.view(-1, 2),
            action_target.view(-1)
        ).view(action_output.shape[:-1])/log(2)
        return (seq_loss, visual_loss, action_loss)

    @torch.no_grad()
    def inference(self, seq_input):
        return self.softmax(self.seq_head(self.model(seq_input)))
