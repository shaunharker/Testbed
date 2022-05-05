import torch
from math import sqrt, log
import random
from torch.nn import Module, Linear, Sequential
from torch.nn import Embedding, LayerNorm
from torch.nn import Sigmoid, ReLU, GELU
from torch.distributions import Categorical
import numpy as np
import chess
from .targets import maketargets, bytes_to_tensor


class CrossEntropyLoss(Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.crossentropyloss = (
            torch.nn.CrossEntropyLoss(reduction='none'))

    def forward(self, x, y):
        return self.crossentropyloss(
            x.view(-1, self.n_classes),
            y.view(-1)).view(x.shape[:-1]
            )/log(2)


class Nonlinearity(Module):
    def __init__(self, **config):
        super().__init__()
        self.nonlinearity = config["nonlinearity"]
        self.f = {"sigmoid": Sigmoid(), "ReLU": ReLU(),
            "GELU": GELU()}[self.nonlinearity]

    def forward(self, x):
        return self.f(x)


class MLP(Module):
    def __init__(self, **config):
        super().__init__()
        m = config["d_model"]
        n = config["d_hidden"]
        self.model = Sequential(
            Linear(m, n, bias=True),
            Nonlinearity(**config),
            Linear(n, m, bias=True))

    def forward(self, x):
        return self.model(x)


class Mask(Module):
    def __init__(self, **config):
        super().__init__()
        self.mask = config["mask"]

    def forward(self, x):
        n, device = x.shape[-1], x.device
        if self.mask == "none":
            return x
        elif self.mask == "causal":
            return x+(1-1/torch.tril(torch.ones((n,n),
                device=device)))


class Attn(Module):
    def __init__(self, **config):
        super().__init__()
        d_model = self.d_model = config["d_model"]
        d_k = self.d_k = config["d_k"]
        d_v = self.d_v = config["d_v"]
        n_heads = self.n_heads = config["n_heads"]
        self.query_proj = Linear(d_model, d_k*n_heads)
        self.key_proj = Linear(d_model, d_k*n_heads)
        self.value_proj = Linear(d_model, d_v*n_heads)
        self.mask = Mask(**config)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.linear = Linear(d_v*n_heads, d_model, bias=False)

    def forward(self, x):
        n_ctx = x.shape[-2]
        split_heads = lambda x: x.view(x.shape[:-1] +
            (self.n_heads, -1)).transpose(-2,
            -3).contiguous()
        merge_heads = lambda x: x.transpose(-2,
            -3).contiguous().view(x.shape[:-3] +
            (n_ctx, self.d_v*self.n_heads))
        (Q, K, V) = map(split_heads,(self.query_proj(x),
            self.key_proj(x), self.value_proj(x)))
        QKT = torch.matmul(Q/sqrt(self.d_k),
            K.transpose(-1,-2))
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
    def __init__(self, **config):
        super().__init__()
        d_model = config["d_model"]
        self.model = Sequential(
            ResidualLayerNorm(Attn(**config), d_model),
            ResidualLayerNorm(MLP(**config), d_model))

    def forward(self, x):
        return self.model(x)


class PositionalEncoding(Module):
    def __init__(self, **config):
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
    def __init__(self, **config):
        super().__init__()
        self.config = {
            "n_classes": 256,
            "n_ctx": 4096,
            "n_layers": 3,
            "plan": [0,1,2],
            "d_model": 4096,
            "d_hidden": 4096,
            "d_k": 64,
            "d_v": 64,
            "n_heads": 64,
            "nonlinearity": "GELU",
            "mask": "causal",
            "device": "cuda"}
        self.config.update(config or dict())
        config = self.config
        n_ctx = config["n_ctx"]
        n_layers = config["n_layers"]
        plan = config["plan"]
        d_model = config["d_model"]
        d_hidden = config["d_hidden"]
        device = config["device"]
        make_layer = lambda: TransformerLayer(**config)
        self.layers = [make_layer() for _ in
            range(n_layers)]
        self.model = Sequential(
            Embedding(256, d_model),
            PositionalEncoding(**config),
            Sequential(*[self.layers[i] for i in plan]))
        self.seq_head = Linear(d_model, 256)
        self.visual_head = Sequential(Linear(d_model,
            64*13), View(64, 13))
        self.action_head = Sequential(Linear(d_model,
            256*2), View(256, 2))
        self.seq_crit = CrossEntropyLoss(256)
        self.visual_crit = CrossEntropyLoss(13)
        self.action_crit = CrossEntropyLoss(2)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.to(device)

    def numel(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, game=None, targets=None):
        # note: seq_target is allowed to be shorter
        (seq_input, seq_target, visual_target,
            action_target) = targets or maketargets(game)
        model_output = self.model(seq_input)
        seq_output = self.seq_head(model_output)
        seq_output = seq_output[...,:seq_target.shape[-1],:]
        visual_output = self.visual_head(model_output)
        action_output = self.action_head(model_output)
        seq_loss = self.seq_crit(seq_output, seq_target)
        visual_loss = self.visual_crit(visual_output, visual_target)
        action_loss = self.action_crit(action_output, action_target)
        return (seq_loss, visual_loss, action_loss)

    @torch.no_grad()
    def inference(self, gamestring):
        seq_input = bytes_to_tensor(gamestring)
        model_output = self.model(seq_input)
        seq_output = self.seq_head(model_output)
        visual_output = self.visual_head(model_output)
        action_output = self.action_head(model_output)
        seq_probs = self.softmax(seq_output)
        visual_probs = self.softmax(visual_output)
        action_probs = self.softmax(action_output)
        return (seq_probs, visual_probs, action_probs)

    def boardstring(self, game, temp=1.0):
        if game == "":
            gamestring = "\n"
        else:
            gamestring = "\n" + game.strip() + " "
        visual_probs = self.inference(gamestring)[1]
        probs = visual_probs[-1]
        result = ""
        pieces = ".KQNBRPkqnbrp"
        for i in range(64):
            result += pieces[Categorical(probs=
                probs[i]**(1.0/temp)).sample().item()
                if temp > 0 else torch.argmax(probs[i])
                .item()]
            if i%8 == 7:
                result += "\n"
        return result

    def move(self, game, temp=1.0):
        if game == "":
            gamestring = "\n"
        else:
            gamestring = "\n" + game.strip() + " "
        board = chess.Board()
        moves = game.split()
        for move in moves:
            board.push_san(move)
        legal = [board.san(move) for move in
            board.legal_moves]
        if len(legal) == 0:
            return None
        if len(legal) == 1:
            return legal[0]
        newmove = ""
        idx = 0
        while True:
            k = len(newmove)
            S = set(ord(move[k]) for move in legal
                if move.startswith(newmove) and
                len(move) > k)
            probs = self.inference(gamestring +
                newmove)[0].view(-1)[-256:] # just [-1]?
            for i in range(256):
                if i not in S:
                    probs[i] = 0
            newmove += chr(Categorical(probs=probs**(
                1.0/temp)).sample().item()
                if temp > 0 else torch.argmax(probs).item())
            left = [move for move in legal
                if move.startswith(newmove)]
            if len(left) == 0:
                return (random.choice(legal) if temp > 0
                    else legal[0])
            if len(left) == 1:
                return left[0]
