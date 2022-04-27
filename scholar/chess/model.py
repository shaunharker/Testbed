import torch
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
        d_model = config["d_model"]
        d_hidden = (lambda x: x if type(x) == list else [x])(config["d_hidden"])
        nonlinearity = config["nonlinearity"]
        self.module = Sequential(
            Linear(d_model, d_hidden[0], bias=True),
            Sequential(
                Sequential(
                    Nonlinearity(nonlinearity),
                    Linear(a, b, bias=True))
                for (a,b) in zip(d_hidden[:-1], d_hidden[1:])),
            Nonlinearity(nonlinearity),
            Linear(d_hidden[-1], d_model, bias=True))

    def forward(self, x):
        return self.module(x)


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
        self.d_model = config["d_model"]
        self.d_k = config["d_k"]
        self.d_v = config["d_v"]
        self.n_heads = config["n_heads"]
        self.query_proj = Linear(self.d_model, self.d_k*self.n_heads)
        self.key_proj = Linear(self.d_model, self.d_k*self.n_heads)
        self.value_proj = Linear(self.d_model, self.d_v*self.n_heads)
        self.mask = Mask(mask=config["mask"])
        self.softmax = torch.nn.Softmax(dim=-1)
        self.linear = Linear(self.d_v*self.n_heads, self.d_model, bias=False)

    def forward(self, x):
        (n_ctx, d_model) = x.shape[-2:]
        assert d_model == self.d_model, f"{d_model} != {self.d_model}"
        split_heads = (lambda x: x.view(x.shape[:-1]+(self.n_heads,
            -1)).transpose(-2,-3).contiguous())
        merge_heads = (lambda x: x.transpose(-2,-3).contiguous().view(
            x.shape[:-3]+(n_ctx,self.d_v*self.n_heads)))
        (Q, K, V) = map(split_heads,(self.query_proj(x), self.key_proj(x),
            self.value_proj(x)))
        QKT = torch.matmul(Q/math.sqrt(self.d_k),K.transpose(-1,-2))
        return self.linear(merge_heads(self.softmax(self.mask(QKT))@V))


class ResidualLayerNorm(Module):
    def __init__(self, layer, d_model):
        super().__init__()
        self.d_model = d_model
        self.layer = layer
        self.layernorm = LayerNorm(d_model)

    def forward(self, x):
        assert x.shape[-1] == self.d_model, f"{x.shape[-1]} != {self.d_model}"
        return self.layernorm(x+self.layer(x))


class TransformerLayer(Module):
    def __init__(self, config):
        super().__init__()
        self.attn = ResidualLayerNorm(Attn(config), config["d_model"])
        self.mlp = ResidualLayerNorm(MLP(config), config["d_model"])

    def forward(self, x):
        return self.mlp(self.attn(x))


class PositionalEncoding(Module):
    def __init__(self, config):
        super().__init__()
        self.n_ctx = config["n_ctx"]
        self.d_model = config["d_model"]
        self.weight = torch.nn.Parameter(0.02*torch.randn(self.n_ctx, self.d_model))

    def forward(self, x):
        n_ctx = x.shape[-2]
        assert n_ctx <= self.n_ctx
        return x + self.weight[:n_ctx]


class ChessLanguageModel(Module):
    def __init__(self, config):
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
            "mask": "causal"}.update(config)
        self.layers = [TransformerLayer(self.config) for _ in range(self.config["n_layers"])]
        self.model = Sequential(
            Embedding(256, d_model),
            PositionalEncoding(n_ctx, d_model),
            Sequential(self.layers[i] for i in self.config["plan"]))
        self.seq_head = Linear(d_model, 256)
        self.visual_head = Linear(d_model, 64*13)
        self.action_head = Linear(d_model, 256*2)
        self.crossentropyloss = torch.nn.CrossEntropyLoss(reduction='none')
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, seq_input, seq_target, visual_target, action_target):
        model_output = self.model(seq_input)
        seq_output = self.seq_head(model_output)
        visual_output = self.visual_head(model_output)
        action_output = self.action_head(model_output)
        # Per seq index, we get a 256 prediction
        seq_loss = self.crossentropyloss(
            seq_output.reshape(-1, 256),
            seq_target.reshape(-1)
        ).view(seq_target.shape[:-1])/math.log(256)
        # Per seq index, we get a 64x13 .pnkqbrPNKQBR
        visual_loss = self.crossentropyloss(
            visual_output.reshape(-1, 13),
            visual_target.reshape(-1)
        ).view(visual_target.shape[:-1])/math.log(13)
        # Per seq index, we get a 256x2 matrix for legal seq outputs
        # where each row can be softmaxed to give probabilities
        action_loss = self.crossentropyloss(
            action_output.reshape(-1, 2),
            action_target.reshape(-1)
        ).view(action_target.shape[:-1])/math.log(2)
        return seq_loss + visual_loss + action_loss

    @torch.no_grad()
    def inference(self, seq_input):
        return self.softmax(self.seq_head(self.model(seq_input)))
