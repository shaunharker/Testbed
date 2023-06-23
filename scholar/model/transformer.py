import math
import torch
import copy
from torch.nn import Module, Linear, LayerNorm, Embedding, ModuleList
from torch.cuda.amp import autocast
from .nn import Sequential, MLP, LanguageModel, ResidualLayerNorm


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
    def __init__(self, d_model, d_k, d_v, n_heads, mask="none"):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.query_proj = Linear(d_model, d_k*n_heads)
        self.key_proj = Linear(d_model, d_k*n_heads)
        self.value_proj = Linear(d_model, d_v*n_heads)
        self.mask = Mask(mask=mask)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.linear = Linear(d_v*n_heads, d_model, bias=False)

    def forward(self, x):
        """
        What happens here?
         x.shape == (bs, n_ctx, d_model)

         This gets transformed by query_proj, key_proj, and value_proj into
         Q', K', V' which are (bs, n_ctx, d_model), but then heads are split,
         and we get
         (bs, n_ctx, n_heads, d_k)
         (bs, n_heads, n_ctx, d_k)
         Now Q as (bs, n_heads, n_ctx, d_k) against KT (bs, n_heads, d_k, n_ctx) gives (bs, n_heads, n_ctx, n_ctx)
         V is going to be (b_s, n_heads, n_ctx, d_v)

         The basic idea here is that x as (bs, n_ctx, d_model) has bs x n_ctx vectors that have the head number
         in the more significant position.
        """
        (n_ctx, d_model) = x.shape[-2:]
        assert d_model == self.d_model, f"{d_model} != {self.d_model}"
        split_heads = lambda x: x.view(x.shape[:-1]+(self.n_heads,-1)).transpose(-2,-3).contiguous()
        merge_heads = lambda x: x.transpose(-2,-3).contiguous().view(x.shape[:-3]+(n_ctx,self.d_v*self.n_heads))
        (Q, K, V) = map(split_heads,(self.query_proj(x),self.key_proj(x),self.value_proj(x)))
        QKT = torch.matmul(Q/math.sqrt(self.d_k),K.transpose(-1,-2))
        return self.linear(merge_heads(self.softmax(self.mask(QKT))@V))


class TransformerLayer(Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_hidden, mask="none"):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_hidden = d_hidden
        self.mask = mask

        # self.attn = ResidualLayerNorm(Attn(d_model, d_k, d_v, n_heads, mask), d_model)
        # self.mlp = ResidualLayerNorm(MLP(d_model, d_hidden, 'GELU', d_model), d_model)
        self.attn = Attn(d_model, d_k, d_v, n_heads, mask)
        self.mlp = MLP(d_model, d_hidden, 'GELU', d_model)

    def forward(self, x):
        return x+self.mlp(x+self.attn(x))


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


class TransformerLMHead(Module):
    def __init__(self,
                 n_vocab_in=256,
                 n_vocab_out=256,
                 n_ctx=4096,
                 d_model=1024,
                 d_k=64,
                 d_v=64,
                 n_heads=16,
                 d_hidden=4096,
                 n_layers=16,
                 pattern=None,
                 read_mode='last',
                 read_pattern=None,
                 mask='causal',
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
        self.mask = mask

        self.transformerlayers = ModuleList(
            TransformerLayer(d_model, d_k, d_v, n_heads, d_hidden, mask)
            for _ in range(n_layers))
        self.embedding = Embedding(n_vocab_in, d_model)
        self.position_encoding = PositionalEncoding(n_ctx, d_model)
        self.read_head = Linear(d_model, n_vocab_out)
        if pattern is None:
            self.pattern = list(range(len(self.transformerlayers)))
        else:
            self.pattern = pattern
        
        self.read_mode = read_mode
        if read_pattern is None:
            if read_mode == 'last':
                self.read_pattern = [self.pattern[-1]]
            elif read_mode == 'all':
                self.read_pattern = self.pattern[:]
        else:
            self.read_pattern = read_pattern
        

    def update_pattern(self, new_pattern):
        self.pattern = new_pattern
        if self.read_mode == 'last':
            self.read_pattern = [new_pattern[-1]]
        elif self.read_mode == 'all':
            self.read_pattern = new_pattern[:]

    # def add_noise(self, noise_level=1e-8):

    #     def fuzz(target):
    #         try:
    #             target.weight.data += noise_level * torch.randn_like(target.weight.data)
    #         except:
    #             pass
    #         try:
    #             target.bias.data += noise_level * torch.randn_like(target.bias.data)
    #         except:
    #             pass

    #     fuzz(self.embedding)
    #     fuzz(self.position_encoding)
    #     fuzz(self.read_head)

    #     # transformer layers
    #     for idx, layer in enumerate(self.transformerlayers):
    #         target = layer.attn.layer
    #         fuzz(target.query_proj)
    #         fuzz(target.key_proj)
    #         fuzz(target.value_proj)
    #         fuzz(target.linear)

    #         target = layer.attn.layernorm
    #         fuzz(target)

    #         target = layer.mlp
    #         fuzz(target.layernorm)
    #         fuzz(target.layer.module.layers[0])
    #         fuzz(target.layer.module.layers[3])
            
    def add_noise(self, noise_level=1e-8):

        def fuzz(target):
            try:
                target.weight.data += noise_level * torch.randn_like(target.weight.data)
            except:
                pass
            try:
                target.bias.data += noise_level * torch.randn_like(target.bias.data)
            except:
                pass

        fuzz(self.embedding)
        fuzz(self.position_encoding)
        fuzz(self.read_head)

        # transformer layers
        for idx, layer in enumerate(self.transformerlayers):
            target = layer.attn
            fuzz(target.query_proj)
            fuzz(target.key_proj)
            fuzz(target.value_proj)
            fuzz(target.linear)

            target = layer.mlp
            fuzz(target.module.layers[0])
            fuzz(target.module.layers[3])


    # increase depth
    def increase_n_layers(self):
        self.transformerlayers.append(TransformerLayer(
            self.d_model,
            self.d_k,
            self.d_v, 
            self.n_heads,
            self.d_hidden,
            self.mask))
        self.n_layers += 1

    # increase width
    def increase_n_heads(self):
        # assume by one
        self.n_heads += 1
        old_d_model = self.d_model
        self.d_model += self.d_k  # assume d_k = d_v; we maintain d_model = n_heads * d_k
        self.d_hidden = 4*self.d_model # maintain this

        def source_to_target(source, target):
            try:
                target.weight.data *= 0.0
                target.weight.data[:source.weight.data.shape[0],:source.weight.data.shape[1]] = source.weight.data
            except:
                pass
            try:
                target.bias.data *= 0.0
                target.bias.data[:source.data.shape[0]] = source.bias.data
            except:
                pass
        
        device = self.embedding.weight.device

        # embedding
        new_embedding = Embedding(self.n_vocab_in, self.d_model).to(device)
        source_to_target(source=self.embedding, target=new_embedding)
        self.embedding = new_embedding

        # positional encoding
        new_position_encoding = PositionalEncoding(self.n_ctx, self.d_model).to(device)
        source_to_target(source=self.position_encoding, target=new_position_encoding)
        self.position_encoding = new_position_encoding

        # read head
        new_read_head = Linear(self.d_model, self.n_vocab_out, bias=True).to(device)
        source_to_target(self.read_head, new_read_head)
        self.read_head = new_read_head

        # transformer layers
        for idx, layer in enumerate(self.transformerlayers):
            new_transformer_layer = TransformerLayer(
                self.d_model,
                self.d_k,
                self.d_v, 
                self.n_heads,
                self.d_hidden,
                self.mask).to(device)
            
            target = new_transformer_layer.attn
            source = layer.attn
            source_to_target(source.query_proj, target.query_proj)
            source_to_target(source.key_proj, target.key_proj)
            source_to_target(source.value_proj, target.value_proj)
            source_to_target(source.linear, target.linear)

            target = new_transformer_layer.mlp
            source = layer.mlp
            source_to_target(source.module.layers[0], target.module.layers[0])
            source_to_target(source.module.layers[3], target.module.layers[3])
            
            self.transformerlayers[idx] = new_transformer_layer

    def forward(self, x):
        with autocast(enabled=self.autocast_enabled):
            x = self.embedding(x)
            x = self.position_encoding(x)
            for idx in self.pattern:
                x = self.transformerlayers[idx](x)
            x = self.read_head(x)
            return x
