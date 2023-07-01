import math
import torch
import copy
from torch.nn import Module, Linear, LayerNorm, Embedding, ModuleList, Sigmoid


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
    def __init__(self, d_in, d_hidden, d_out, d_k, d_v, n_heads):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn = Attn(d_in, d_hidden, d_k, d_v, n_heads, 'causal')
        self.nonlinearity = Sigmoid()
        self.ff = Linear(d_hidden, d_out, bias=True)

        self.residual = Linear(d_in, d_out, bias=False)
        self.residual.weight.data *= 0.0
        rows = torch.arange(max(d_in, d_out)) % d_in
        cols = torch.arange(max(d_in, d_out)) % d_out
        self.residual.weight.data[rows, cols] = 1.0


    def forward(self, x):
        return self.ff(self.nonlinearity(self.attn(x))) + self.residual(x)
    
class Persephone(Module):
    def __init__(self,
                 n_vocab_in=256,
                 n_vocab_out=256,
                 n_ctx=4096,
                 d_embd=1024):
        super().__init__()
        self.n_vocab_in = n_vocab_in
        self.n_vocab_out = n_vocab_out
        self.n_ctx = n_ctx
        self.d_embd = d_embd

        self.embedding = Embedding(n_vocab_in, d_embd)
        self.positional_encoding = Embedding(n_ctx, d_embd)
        self.positional_encoding.weight.data = 0.02*torch.randn(n_ctx, d_embd)
        self.layers = ModuleList()
        self.read_heads = ModuleList()


    def add_layer(self, d_in=None, d_hidden=None, d_out=None, d_k=None, d_v=None, n_heads=None):
        device = self.embedding.weight.device # arbitrarily chosen parameter to read a 'device' copy
        self.layers.append(PersephoneLayer(d_in, d_hidden, d_out, d_k, d_v, n_heads).to(device))
        self.read_heads.append(Linear(d_out, self.n_vocab_out).to(device))

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding.weight[:x.shape[-2],:]
        ys = []
        for (layer, read_head) in zip(self.layers, self.read_heads):
            x = layer(x)
            y = read_head(x)
            ys.append(y)
        return torch.stack(ys, dim=0)

    def get_config(self):
        return {
            'n_vocab_in': self.n_vocab_in,
            'n_vocab_out': self.n_vocab_out,
            'n_ctx': self.n_ctx,
            'd_embd': self.d_embd,
        }  
    
    def set_config(self, config):
        self.n_vocab_in = config.get('n_vocab_in', self.n_vocab_in)
        self.n_vocab_out = config.get('n_vocab_out', self.n_vocab_out)
        self.n_ctx = config.get('n_ctx', self.n_ctx)
        self.d_embd = config.get('d_embd', self.d_embd)

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
