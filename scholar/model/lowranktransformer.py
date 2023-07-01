import math
import torch
import copy
from torch.nn import Module, Linear, LayerNorm, Embedding, ModuleList, GELU


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


class LowRankLinear(Module):
    def __init__(self, d_in, d_rank, d_out, bias):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_rank = d_rank
        self.linear1 = Linear(d_in, d_rank, bias=False)
        self.linear2 = Linear(d_rank, d_out, bias=bias)

    def forward(self, x):
        return self.linear2(self.linear1(x))
    
    def update(self, d_rank):
        A = Linear(self.d_in, d_rank, bias=False)
        B = Linear(d_rank, self.d_out, bias=True)
        A.weight.data *= 0.0
        A.weight.data[:,:self.d_rank] = self.linear1.weight.data
        B.weight.data *= 0.0
        B.weight.data[:self.d_rank,:] = self.linear2.weight.data
        B.bias.data = self.linear2.bias.data
        self.d_rank = d_rank 
        self.linear1 = A
        self.linear2 = B


class Attn(Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_rank, mask="causal"):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        
        self.mask = Mask(mask=mask)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.query_proj = LowRankLinear(d_model, d_rank, d_k*n_heads, bias=True)
        self.key_proj = LowRankLinear(d_model, d_rank, d_k*n_heads, bias=True)
        self.value_proj = LowRankLinear(d_model, d_rank, d_v*n_heads, bias=True)
        self.linear = LowRankLinear(d_v*n_heads, d_rank, d_model, bias=False)


    def forward(self, x):
        (n_ctx, d_model) = x.shape[-2:]
        assert d_model == self.d_model, f"{d_model} != {self.d_model}"
        split_heads = lambda x: x.view(x.shape[:-1]+(self.n_heads,-1)).transpose(-2,-3).contiguous()
        merge_heads = lambda x: x.transpose(-2,-3).contiguous().view(x.shape[:-3]+(n_ctx,self.d_v*self.n_heads))
        (Q, K, V) = map(split_heads,(self.query_proj(x),self.key_proj(x),self.value_proj(x)))
        QKT = torch.matmul(Q/math.sqrt(self.d_k),K.transpose(-1,-2))
        return self.linear(merge_heads(self.softmax(self.mask(QKT))@V))
    
    def update(self, d_rank):
        self.d_rank = d_rank
        self.query_proj.update(d_rank)
        self.key_proj.update(d_rank)
        self.value_proj.update(d_rank)
        self.linear.update(d_rank)


class PersephoneLayer(Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_hidden, d_rank, use_layernorms=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_hidden = d_hidden
        self.d_rank = d_rank

        self.use_layernorms = use_layernorms

        if use_layernorms:
            self.ln1 = LayerNorm(d_model)
            self.ln2 = LayerNorm(d_model)
        self.attn = Attn(d_model, d_k, d_v, n_heads, d_rank, 'causal')
        self.ff1 = LowRankLinear(d_model, d_rank, d_hidden, bias=True)
        self.ff2 = LowRankLinear(d_hidden, d_rank, d_model, bias=True)
        self.nonlinearity = GELU()

        self.mlp = lambda x: self.ff2(self.nonlinearity(self.ff1(x)))

    def forward(self, x):
        if self.use_layernorms:
            return x + self.ln2(self.mlp(x + self.ln1(self.attn(x))))
        else:
            return x+self.mlp(x+self.attn(x))

    def update(self, d_rank):
        self.d_rank = d_rank
        self.attn.update(d_rank)
        self.ff1.update(d_rank)
        self.ff2.update(d_rank)
    
class Persephone(Module):
    def __init__(self,
                 n_vocab_in=256,
                 n_vocab_out=256,
                 n_ctx=4096,
                 d_model=1024,
                 d_k=64,
                 d_v=64,
                 n_heads=16,
                 d_hidden=4096,
                 architecture=None,
                 pattern=None,
                 use_layernorms=True,
                 n_layers=16):
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
        self.pattern = pattern
        self.use_layernorms = use_layernorms

        self.embedding = Embedding(n_vocab_in, d_model)
        self.positional_encoding = Embedding(n_ctx, d_model)
        architecture = architecture if architecture else ([16, 32, 64, 128, 256, 512, 1024] + [1024]*(n_layers))
        architecture = architecture[:n_layers]
        self.architecture = architecture
        layer = lambda idx: PersephoneLayer(d_model, d_k, d_v, n_heads, d_hidden, d_rank=architecture[idx], use_layernorms=use_layernorms)
        self.layers = ModuleList(layer(idx) for idx in range(n_layers))
        self.read_heads = ModuleList(Linear(d_model, n_vocab_out) for _ in range(n_layers))

    def update(self, architecture):
        self.architecture = architecture
        for idx, d_rank in architecture:
            self.layers[idx].update(d_rank)

    
    def forward(self, x):
        pattern = self.pattern if self.pattern else list(range(self.n_layers))
        x = self.embedding(x)
        x = x + self.positional_encoding.weight[:x.shape[-2],:]
        ys = []
        for idx in pattern:
            x = self.layers[idx](x)
            y = self.read_heads[idx](x)
            ys.append(y)
        return torch.stack(ys, dim=0)

    def get_config(self):
        return {
            'n_vocab_in': self.n_vocab_in,
            'n_vocab_out': self.n_vocab_out,
            'n_ctx': self.n_ctx,
            'd_model': self.d_model,
            'd_k': self.d_k,
            'd_v': self.d_k,
            'n_heads': self.n_heads,
            'd_hidden': self.d_hidden,
            'n_layers': self.n_layers,
            'use_layernorms': self.use_layernorms,
        }  
    
    def set_config(self, config):
        self.n_vocab_in = config.get('n_vocab_in', self.n_vocab_in)
        self.n_vocab_out = config.get('n_vocab_out', self.n_vocab_out)
        self.n_ctx = config.get('n_ctx', self.n_ctx)
        self.d_model = config.get('d_model', self.d_model)
        self.d_k = config.get('d_k', self.d_k)
        self.d_v = config.get('d_v', self.d_v)
        self.n_heads = config.get('n_heads', self.n_heads)
        self.d_hidden = config.get('d_hidden', self.d_hidden)
        self.n_layers = config.get('n_layers', self.n_layers)
        self.use_layernorms = config.get('use_layernorms', self.use_layernorms)

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.get_config()
        }, f=path)

    @staticmethod
    def load(path):
        checkpoint = torch.load(path)
        config = checkpoint.get('config', {})
        model = Persephone(**config)  # replace with your actual model constructor
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
