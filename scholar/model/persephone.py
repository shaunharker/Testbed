import math
import torch
import random
from torch.nn import Module, Linear, Embedding, ModuleList, LayerNorm, GELU


class Mask(Module):
    def __init__(self, mask="none"):
        super().__init__()
        self.mask = mask

    def forward(self, x):
        n, device = x.shape[-1], x.device
        if self.mask == "none":
            return x
        elif self.mask == "causal":
            weight = (1-1/torch.tril(torch.ones((n,n),device=device)))
            return x + weight


class Attn(Module):
    def __init__(self, d_in, d_out, d_k, d_v, n_heads, mask="causal"):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        
        self.mask = Mask(mask=mask)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.query_proj = Linear(d_in, d_k*n_heads, bias=True)
        self.key_proj = Linear(d_in, d_k*n_heads, bias=True)
        self.value_proj = Linear(d_in, d_v*n_heads, bias=True)
        self.linear = Linear(d_v*n_heads, d_out, bias=True)

    def forward(self, x):
        (n_ctx, d_model) = x.shape[-2:]
        assert d_model == self.d_in, f"{d_model} != {self.d_in}"
        split_heads = lambda x: x.view(x.shape[:-1]+(self.n_heads,-1)).transpose(-2,-3).contiguous()
        merge_heads = lambda x: x.transpose(-2,-3).contiguous().view(x.shape[:-3]+(n_ctx,self.d_v*self.n_heads))
        (Q, K, V) = map(split_heads,(self.query_proj(x),self.key_proj(x),self.value_proj(x)))
        QKT = torch.matmul(Q/math.sqrt(self.d_k),K.transpose(-1,-2))
        return self.linear(merge_heads(self.softmax(self.mask(QKT))@V))


class PersephoneLayer(Module):
    def __init__(self, d_model, d_hidden, d_k, d_v, n_heads):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn = Attn(d_model, d_model, d_k, d_v, n_heads, 'causal')
        self.ln1 = LayerNorm(d_model)
        self.ff1 = Linear(d_model, d_hidden)
        self.nonlinearity = GELU()
        self.ff2 = Linear(d_hidden, d_model)
        self.ln2 = LayerNorm(d_model)
        self.mlp = lambda x: self.ff2(self.nonlinearity(self.ff1(x)))
    def forward(self, x):
        return x + self.ln2(self.mlp(x + self.ln1(self.attn(x))))
        # x1 = x + self.ln1(self.attn(x))
        # x2 = x1 + self.ln2(self.mlp(x1))
        # return x2

class Persephone(Module):
    def __init__(self,
                 n_vocab_in=256,
                 n_vocab_out=256,
                 n_ctx=4096,
                 d_embd=8,
                 architecture=None):
        super().__init__()
        self.n_vocab_in = n_vocab_in
        self.n_vocab_out = n_vocab_out
        self.n_ctx = n_ctx
        self.d_embd = d_embd
        self.architecture = architecture if architecture is not None else []

        self.d_model = d_embd # stores the dimensionality of last layer, which starts and embedding dim.

        self.embedding = Embedding(self.n_vocab_in, self.d_embd)
        self.positional_encoding = Embedding(self.n_ctx, self.d_embd)
        self.positional_encoding.weight.data *= 0.02
        self.connectors = ModuleList()
        self.layers = ModuleList()
        self.read_heads = ModuleList()
        for (d_model, d_hidden, d_k, d_v, n_heads) in self.architecture:
            self.add_layer(d_model, d_hidden, d_k, d_v, n_heads, False)


    def forward(self, input_ids):
        device = input_ids.device
        positions = torch.arange(input_ids.size(-1), device=device)
        ys = []
        x = self.embedding(input_ids) + self.positional_encoding(positions)
        for connector, layer, read_head in zip(self.connectors, self.layers, self.read_heads):
            x = connector(x)
            x = layer(x)
            y = read_head(x)
            ys.append(y)
        return torch.stack(ys, dim=0)
    
    def add_layer(self, d_model=None, d_hidden=None, d_k=None, d_v=None, n_heads=None, add_to_architecture=True):
        if d_model is None:
            d_model = self.d_model
        if d_k is None:
            if n_heads is not None:
                d_k = d_model // n_heads
            else:
                if d_model < 1024:
                    try:
                        d_k = {1:1, 2:2, 4:4, 8:4, 16:8, 32:8, 64:16, 128:16, 256:32, 512:32}[d_model]
                    except:
                        d_k = min(64, d_model)
                else:
                    d_k = 64
        if d_v is None:
            d_v = d_k
        if n_heads is None:
            n_heads = d_model // d_k
        if d_hidden is None:
            d_hidden = 4*d_model
        device = self.embedding.weight.device
        
        # issue: when d_model does not change, isn't linear connector superfluous?
        self.connectors.append(Linear(self.d_model, d_model).to(device))
        self.layers.append(PersephoneLayer(d_model, d_hidden, d_k, d_v, n_heads).to(device))
        self.read_heads.append(Linear(d_model, self.n_vocab_out).to(device))
        if add_to_architecture:
            self.architecture.append((d_model, d_hidden, d_k, d_v, n_heads))
        self.d_model = d_model

    def get_config(self):
        return {
            'n_vocab_in': self.n_vocab_in,
            'n_vocab_out': self.n_vocab_out,
            'n_ctx': self.n_ctx,
            'd_embd': self.d_embd,
            'architecture': self.architecture
        }  


    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.get_config()
        }, f=path)

    @staticmethod
    def load(path):
        checkpoint = torch.load(path)
        config = checkpoint['config']
        sd = checkpoint['state_dict']
        module = Persephone(**config)
        module.load_state_dict(sd)
        return module
