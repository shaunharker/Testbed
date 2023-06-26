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
        #return self.mlp(self.attn(x))
        return x+self.mlp(x+self.attn(x))


class LowRankPositionalEncoding(Module):
    def __init__(self, n_ctx, d_pos, d_model):
        super().__init__()
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.d_pos = d_pos
        self.weight = torch.nn.Parameter(0.02*torch.randn(n_ctx, d_pos))
        self.linear = Linear(d_pos, d_model) #MLP(d_pos, 2*d_model, 'GELU', d_model)

    def forward(self, x):
        n_ctx = x.shape[-2]
        assert n_ctx <= self.n_ctx
        return x + self.linear(self.weight[:n_ctx])

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
                 d_pos=16,
                 pattern=None,
                 mask='causal',
                 positional_encoding_mode='standard',
                 read_mode='last',
                 read_head_type='linear',
                 autocast_enabled=None):
        super().__init__()
        self.n_vocab_in = n_vocab_in
        self.n_vocab_out = n_vocab_out
        self.n_ctx = n_ctx
        self.d_pos = d_pos # dimensionality of position encoding
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

        self.positional_encoding_mode = positional_encoding_mode
        if positional_encoding_mode == 'standard':
            self.positional_encoding = PositionalEncoding(n_ctx, d_model)
        elif positional_encoding_mode == 'lowrank':
            self.positional_encoding = LowRankPositionalEncoding(n_ctx, d_pos, d_model)
        
        self.read_head_type = read_head_type
        if read_head_type == 'linear':
            self.read_head = Linear(d_model, n_vocab_out)
        elif read_head_type == 'mlp':
            self.read_head =  MLP(d_model, d_hidden, 'GELU', n_vocab_out)

        if pattern is None:
            self.pattern = list(range(len(self.transformerlayers)))
        else:
            self.pattern = pattern
        
        self.read_mode = read_mode
    
    def forward(self, x):
        with autocast(enabled=self.autocast_enabled):
            x = self.embedding(x)
            x = self.positional_encoding(x)
            ys = []
            for idx in self.pattern:
                x = self.transformerlayers[idx](x)
                if self.read_mode == 'all':
                    y = self.read_head(x)
                    ys.append(y)
            if self.read_mode == 'last':
                x = self.read_head(x)
                return x
            else:
                # Stack all the outputs along the 0-th dimension
                # resulting shape will be: (n_layers, bs, n_ctx, n_vocab_out)
                return torch.stack(ys, dim=0)

    def update_pattern(self, new_pattern):
        self.pattern = new_pattern

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
        fuzz(self.positional_encoding)
        if self.positional_encoding_mode == 'standard':
            fuzz(self.positional_encoding)
        elif self.positional_encoding_mode == 'lowrank':
            fuzz(self.positional_encoding.linear)

        if self.read_head_type == 'linear':
            fuzz(self.read_head)
        elif self.read_head_type == 'mlp':
            pass  # worry about later

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
        device = self.embedding.weight.device
        self.transformerlayers.append(TransformerLayer(
            self.d_model,
            self.d_k,
            self.d_v, 
            self.n_heads,
            self.d_hidden,
            self.mask).to(device))
        self.n_layers += 1
        self.pattern.append(len(self.transformerlayers)-1)

    def double_heads(self):
         # double n_heads, d_model, and d_hidden, duplicating weight data and dividing by 2 where appropriate.
        self.n_heads *= 2
        old_d_model = self.d_model
        self.d_model *= 2 # assume d_k = d_v; we maintain d_model = n_heads * d_k
        self.d_hidden *= 2

        def source_to_target(source, target):
            weight_exists = False
            bias_exists = False
            
            try:
                s = target.weight.shape
                weight_exists = True
            except:
                pass
            
            try:
                s = target.bias.shape
                bias_exists = True
            except:
                pass
            
            if weight_exists:
                multiple1 = target.weight.shape[0] // source.weight.shape[0]
                multiple2 = target.weight.shape[1] // source.weight.shape[1]
                print(source.weight.shape, target.weight.shape)
                source.weight.repeat(multiple1,multiple2)
                target.weight.data.copy_(source.weight.data.repeat(multiple1,multiple2))
                if multiple1 == 2:
                    target.weight.data *= .5 # keep it equivalent with two copies of the inputs coming in 

            if bias_exists:
                print(source.bias.shape, target.bias.shape)
                multiple1 = target.bias.shape[0] // source.bias.shape[0]
                target.bias.data.copy_(source.bias.data.repeat(multiple1))

        device = self.embedding.weight.device

        # embedding
        print("Embedding.")
        new_embedding = Embedding(self.n_vocab_in, self.d_model).to(device)
        source_to_target(source=self.embedding, target=new_embedding)
        self.embedding = new_embedding

        # positional encoding
        print("Positional Encoding.")
        if self.positional_encoding_mode == 'standard':
            new_positional_encoding = PositionalEncoding(self.n_ctx, self.d_model).to(device)
            source_to_target(source=self.positional_encoding, target=new_positional_encoding)
            self.positional_encoding = new_positional_encoding
        if self.positional_encoding_mode == 'lowrank':
            new_positional_encoding = LowRankPositionalEncoding(self.n_ctx, self.d_pos, self.d_model).to(device)
            source_to_target(source=self.positional_encoding, target=new_positional_encoding)
            source_to_target(source=self.positional_encoding.linear, target=new_positional_encoding.linear)
            self.positional_encoding = new_positional_encoding

        # read head
        print("Read Head.")
        new_read_head = Linear(self.d_model, self.n_vocab_out, bias=True).to(device)
        source_to_target(self.read_head, new_read_head)
        self.read_head = new_read_head

        # transformer layers
        print("Transformer layers.")
        for idx, layer in enumerate(self.transformerlayers):
            print(f"Layer {idx}")
            new_transformer_layer = TransformerLayer(
                self.d_model,
                self.d_k,
                self.d_v, 
                self.n_heads,
                self.d_hidden,
                self.mask).to(device)
            
            target = new_transformer_layer.attn
            source = layer.attn
            print("query")
            source_to_target(source.query_proj, target.query_proj)
            print("key")
            source_to_target(source.key_proj, target.key_proj)
            print("value")
            source_to_target(source.value_proj, target.value_proj)
            print("linear")
            source_to_target(source.linear, target.linear)

            target = new_transformer_layer.mlp
            source = layer.mlp
            print("ff1")
            source_to_target(source.module.layers[0], target.module.layers[0])
            print("ff2")
            source_to_target(source.module.layers[3], target.module.layers[3])
            
            self.transformerlayers[idx] = new_transformer_layer


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
        new_positional_encoding = PositionalEncoding(self.n_ctx, self.d_pos, self.d_model).to(device)
        source_to_target(source=self.positional_encoding, target=new_positional_encoding)
        source_to_target(source=self.positional_encoding.linear, target=new_positional_encoding.linear)
        self.positional_encoding = new_positional_encoding

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


