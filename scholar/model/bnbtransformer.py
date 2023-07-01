import math
import torch
import copy
from torch.nn import Module, Linear, LayerNorm, Embedding, ModuleList
from torch.cuda.amp import autocast
from .nn import Sequential, MLP, LanguageModel, ResidualLayerNorm



class Mask(Module):
    def __init__(self, mask="none", use_bitsandbytes=False):
        super().__init__()
        self.mask = mask
        self.use_bitsandbytes = use_bitsandbytes

    def forward(self, x):
        n, device = x.shape[-1], x.device
        if self.mask == "none":
            return x
        elif self.mask == "causal":
            weight = (1-1/torch.tril(torch.ones((n,n),device=device)))
            if self.use_bitsandbytes:   
                weight = Int8Params(weight, requires_grad=False, has_fp16_weights=False)
            return x + weight


class Attn(Module):
    def __init__(self, d_model, d_k, d_v, n_heads, mask="none", use_bitsandbytes=False):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        
        self.mask = Mask(mask=mask, use_bitsandbytes=use_bitsandbytes)
        self.softmax = torch.nn.Softmax(dim=-1)

        if use_bitsandbytes:
            make = lambda m, n, bias: bnb.nn.Linear8bitLt(m, n, bias=bias, has_fp16_weights=False, threshold=6.0)
        else:
            make = lambda m, n, bias: Linear(m, n, bias=bias)

        self.query_proj = make(d_model, d_k*n_heads, True)
        self.key_proj = make(d_model, d_k*n_heads, True)
        self.value_proj = make(d_model, d_v*n_heads, True)
        self.linear = make(d_v*n_heads, d_model, False)


    def forward(self, x):
        (n_ctx, d_model) = x.shape[-2:]
        assert d_model == self.d_model, f"{d_model} != {self.d_model}"
        split_heads = lambda x: x.view(x.shape[:-1]+(self.n_heads,-1)).transpose(-2,-3).contiguous()
        merge_heads = lambda x: x.transpose(-2,-3).contiguous().view(x.shape[:-3]+(n_ctx,self.d_v*self.n_heads))
        (Q, K, V) = map(split_heads,(self.query_proj(x),self.key_proj(x),self.value_proj(x)))
        QKT = torch.matmul(Q/math.sqrt(self.d_k),K.transpose(-1,-2))
        return self.linear(merge_heads(self.softmax(self.mask(QKT))@V))


class TransformerLayer(Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_hidden, mask="none", use_layernorms=True, use_bitsandbytes=False):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_hidden = d_hidden
        self.mask = mask
        self.use_bitsandbytes = use_bitsandbytes

        # if use_bitsandbytes:
        #     use_layernorms = False # TODO: implement bnb LayerNorm

        self.use_layernorms = use_layernorms

        if use_layernorms:
            self.ln1 = LayerNorm(d_model)
            self.ln2 = LayerNorm(d_model)
        self.attn = Attn(d_model, d_k, d_v, n_heads, mask, use_bitsandbytes)
        self.mlp = MLP(d_model, d_hidden, 'GELU', d_model, use_bitsandbytes)

    def forward(self, x):
        if self.use_layernorms:
            return x + self.ln2(self.mlp(x + self.ln1(self.attn(x))))
        else:
            return x+self.mlp(x+self.attn(x))


class LowRankPositionalEncoding(Module):
    def __init__(self, n_ctx, d_pos, d_model, use_bitsandbytes):
        super().__init__()
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.d_pos = d_pos
        self.use_bitsandbytes = use_bitsandbytes
        self.weight = torch.nn.Parameter(0.02*torch.randn(n_ctx, d_pos))
        if use_bitsandbytes:
            self.weight = Int8Params(self.weight, has_fp16_weights=False)
            self.linear = bnb.nn.Linear8bitLt(d_pos, d_model, bias=True, has_fp16_weights=False, threshold=6.0)
        else:
            self.linear = Linear(d_pos, d_model) #MLP(d_pos, 2*d_model, 'GELU', d_model)

    def forward(self, x):
        n_ctx = x.shape[-2]
        assert n_ctx <= self.n_ctx
        return x + self.linear(self.weight[:n_ctx])

class PositionalEncoding(Module):
    def __init__(self, n_ctx, d_model, use_bitsandbytes):
        super().__init__()
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.use_bitsandbytes = use_bitsandbytes 
        if use_bitsandbytes:
            self.embedding = bnb.nn.StableEmbedding(n_ctx, d_model)
        else:
            self.weight = torch.nn.Parameter(0.02*torch.randn(n_ctx, d_model))
    def forward(self, x):
        n_ctx = x.shape[-2]
        assert n_ctx <= self.n_ctx
        if self.use_bitsandbytes:
            weight = self.embedding.weight
        else:
            weight = self.weight
        return x + weight[:n_ctx]
    
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
                 use_layernorms=True,
                 use_bitsandbytes=True,
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
        self.use_layernorms = use_layernorms
        self.use_bitsandbytes = use_bitsandbytes

        self.transformerlayers = ModuleList(
            TransformerLayer(d_model, d_k, d_v, n_heads, d_hidden, mask, use_layernorms, use_bitsandbytes)
            for _ in range(n_layers))
        
        if use_bitsandbytes:
            self.embedding = bnb.nn.StableEmbedding(n_vocab_in, d_model)
        else:
            self.embedding = Embedding(n_vocab_in, d_model)

        self.positional_encoding_mode = positional_encoding_mode
        if positional_encoding_mode == 'standard':
            self.positional_encoding = PositionalEncoding(n_ctx, d_model, use_bitsandbytes)
        elif positional_encoding_mode == 'lowrank':
            self.positional_encoding = LowRankPositionalEncoding(n_ctx, d_pos, d_model, use_bitsandbytes)
        
        self.read_head_type = read_head_type
        if read_head_type == 'linear':
            if use_bitsandbytes:
                self.read_head = bnb.nn.Linear8bitLt(d_model, n_vocab_out, bias=True, has_fp16_weights=False, threshold=6.0)
            else:
                self.read_head = Linear(d_model, n_vocab_out)
        elif read_head_type == 'mlp':
            self.read_head =  MLP(d_model, d_hidden, 'GELU', n_vocab_out, use_bitsandbytes)

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

    def get_config(self):
        return {
            'n_vocab_in': self.n_vocab_in,
            'n_vocab_out': self.n_vocab_out,
            'n_ctx': self.n_ctx,
            'd_pos': self.d_pos,
            'd_model': self.d_model,
            'd_k': self.d_k,
            'd_v': self.d_k,
            'n_heads': self.n_heads,
            'd_hidden': self.d_hidden,
            'n_layers': self.n_layers,
            'positional_encoding_mode': self.positional_encoding_mode,
            'read_mode': self.read_mode,
            'read_head_type': self.read_head_type,
            'mask': self.mask,
            'use_layernorms': self.use_layernorms,
            'use_bitsandbytes': self.use_bitsandbytes,
        }  
    
    def set_config(self, config):
        self.n_vocab_in = config.get('n_vocab_in', self.n_vocab_in)
        self.n_vocab_out = config.get('n_vocab_out', self.n_vocab_out)
        self.n_ctx = config.get('n_ctx', self.n_ctx)
        self.d_pos = config.get('d_pos', self.d_pos)
        self.d_model = config.get('d_model', self.d_model)
        self.d_k = config.get('d_k', self.d_k)
        self.d_v = config.get('d_v', self.d_v)
        self.n_heads = config.get('n_heads', self.n_heads)
        self.d_hidden = config.get('d_hidden', self.d_hidden)
        self.n_layers = config.get('n_layers', self.n_layers)
        self.use_layernorms = config.get('use_layernorms', self.use_layernorms)
        self.use_bitsandbytes = config.get('use_bitsandbytes', self.use_bitsandbytes)


    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.get_config()
        }, f=path)

    @staticmethod
    def load(path):
        checkpoint = torch.load(path)
        config = checkpoint.get('config', {})
        model = TransformerLMHead(**config)  # replace with your actual model constructor
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    

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

        def source_to_target(source, target, halfweight=False):
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
                if halfweight:
                    target.weight.data *= 0.5

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
        source_to_target(self.read_head, new_read_head, halfweight=True)
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
            source_to_target(source.query_proj, target.query_proj, halfweight=True)
            print("key")
            source_to_target(source.key_proj, target.key_proj, halfweight=True)
            print("value")
            source_to_target(source.value_proj, target.value_proj, halfweight=True)
            print("linear")
            source_to_target(source.linear, target.linear, halfweight=True)

            target = new_transformer_layer.mlp
            source = layer.mlp
            print("ff1")
            source_to_target(source.module.layers[0], target.module.layers[0], halfweight=True)
            print("ff2")
            source_to_target(source.module.layers[3], target.module.layers[3], halfweight=True)
            
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


