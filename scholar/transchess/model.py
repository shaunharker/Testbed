import torch
from math import sqrt, log
from torch.nn import Module, Linear, Sequential, Embedding, LayerNorm, Sigmoid, ReLU, GELU
import numpy as np

bytes_to_tensor = lambda x: torch.tensor(np.frombuffer(
    bytes(x, encoding='utf8'), dtype=np.uint8),
    dtype=torch.long, device="cuda")


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
            "n_layers": 9,
            "plan": [0,1,2,3,4,5,6,7,8],
            "d_model": 2500,
            "d_k": 50,
            "d_v": 50,
            "n_heads": 50,
            "d_hidden": 2500,
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

    def numel(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, game):
        (seq_input, seq_target, visual_target, action_target) = targets(game)
        model_output = self.model(seq_input)
        seq_output = self.seq_head(model_output)
        visual_output = self.visual_head(model_output)
        action_output = self.action_head(model_output)
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
        return (game, seq_input, seq_target, visual_target, action_target, seq_loss, visual_loss, action_loss)

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

    def move(self, game, temp=1.0):
        Categorical = torch.distributions.Categorical
        if game == "":
            gamestring = "\n"
        else:
            gamestring = "\n" + game.strip() + " "
        move = ""
        while True:
            (probs, _, _) = self.inference(gamestring)
            probs = probs.view(-1)[-256:]
            if temp > 0:
                y = Categorical(probs=
                    probs**(1.0/temp)).sample().item()
            else:
                y = torch.argmax(probs).item()
            if y == 32:
                break
            if y == 10:
                break
            move += chr(y)
            gamestring += chr(y)
            if len(move) > 8:
                break
        return move

import chess

def targets(game):
    def look(board):
        piece_encoding = {'.': 0, 'K': 1, 'Q': 2, 'N': 3, 'B': 4, 'R': 5,
            'P': 6, 'k': 7, 'q': 8, 'n': 9, 'b': 10, 'r': 11, 'p': 12}
        fen = (board.fen().split()[0].replace("/", '').replace("8", "44")
            .replace("7", "43").replace("6", "33").replace("5", "32")
            .replace("4", "22").replace("3", "21").replace("2", "11")
            .replace("1", "."))
        return torch.tensor([piece_encoding[c] for c in fen],
            dtype=torch.long,  device="cuda")

    def chunk(move, legal):
        n = len(move) + 1
        action_target_chunk = torch.zeros([n, 256], dtype=torch.long,
            device="cuda")
        for d in range(len(move)):
            c = move[d]
            for m in legal:
                if len(m) > d:
                    action_target_chunk[d, ord(m[d])] = 1
            legal = [m for m in legal if len(m) > d and m[d] == c]
        return action_target_chunk

    moves = game.split()
    N = len(game.strip()) + 1 # add one for newline preamble
    visual_target = torch.zeros([N,64], dtype=torch.long, device="cuda")
    action_target = torch.zeros([N,256], dtype=torch.long, device="cuda")
    board = chess.Board()
    idx = 0
    legal = [board.san(m) for m in board.legal_moves]
    for move in moves:
        n = len(move) + 1
        visual_target[idx:idx+n,:] = look(board).reshape([1,-1])
        action_target[idx:idx+n,:] = chunk(move, legal)
        board.push_san(move)
        legal = [board.san(m) for m in board.legal_moves]
        if len(legal) > 0:
            action_target[idx+n-1, ord(" ")] = 1
        else:
            action_target[idx+n-1, ord("\n")] = 1
        idx += n

    seq_input = bytes_to_tensor("\n" + game.strip())
    seq_target = bytes_to_tensor(game.strip() +
        ("\n" if len(legal)==0 else " "))
    return seq_input, seq_target, visual_target, action_target
