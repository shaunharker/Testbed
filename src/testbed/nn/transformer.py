import math
import torch
import torch.nn.functional as F
from time import time
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Dropout, Embedding, Linear, CrossEntropyLoss, Softmax, LayerNorm, ModuleList
from torch.cuda.amp import autocast

class MyLayerNorm(LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy = 0.0
        self.compute_time = 0.0

    @autocast()
    def forward(self, x):
        start_time = time()
        result = super().forward(x)
        torch.cuda.synchronize()
        self.energy += 3.0 * 6.0 * torch.numel(x) # ?
        self.compute_time += time() - start_time
        return result

    def compute_data(self):
        return {"name": "MyLayerNorm",
                "time": self.compute_time,
                "energy": self.energy}

class MyLinear(Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features=in_features,
                         out_features=out_features)
        self.energy = 0.0
        self.compute_time = 0.0

    @autocast()
    def forward(self, x):
        start_time = time()
        result = super().forward(x)
        torch.cuda.synchronize()
        self.energy += self.out_features * torch.numel(x)
        self.compute_time += time() - start_time
        return result

    def compute_data(self):
        return {"name": "MyLinear",
                "time": self.compute_time,
                "energy": self.energy}

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        d_head = d_k = d_v = d_model // n_heads # assume these are equal for this implementation
        self.d_head = d_head
        self.layernorm = MyLayerNorm(d_model)
        self.query_projection = MyLinear(d_model, d_k * n_heads)
        self.key_projection = MyLinear(d_model, d_k * n_heads)
        self.value_projection = MyLinear(d_model, d_v * n_heads)
        self.output_projection = MyLinear(d_v * n_heads, d_model)
        self.compute_time = 0.0
        self.energy = 0.0
        self.data = {"preamble": {"time": 0.0, "energy": 0.0},
                     "split_heads": {"time": 0.0, "energy": 0.0},
                     "matmul(Q,K^T)": {"time": 0.0, "energy": 0.0},
                     "compute_attention": {"time": 0.0, "energy": 0.0},
                     "matmul(A,V)": {"time": 0.0, "energy": 0.0},
                     "merge_heads": {"time": 0.0, "energy": 0.0}}
    def compute_data(self):
        data = {"name": "MultiHeadSelfAttention",
                "time": self.compute_time,
                "data": self.data,
                "children": {
                    "layernorm": self.layernorm.compute_data(),
                    "query_projection": self.query_projection.compute_data(),
                    "key_projection": self.key_projection.compute_data(),
                    "value_projection": self.value_projection.compute_data(),
                    "output_projection": self.output_projection.compute_data()}}
        data["self_time"] = data["time"] - sum(data["children"][key]["time"]
                                               for key in data["children"])
        data["self_time2"] = sum(data["data"][key]["time"] for key in data["data"])
        data["energy"] = (sum(data["data"][key]["energy"] for key in data["data"])+
                          sum(data["children"][key]["energy"] for key in data["children"]))
        return data

    @autocast()
    def forward(self, X):
        """
        input: X has shape [..., n_ctx, d_model]
        output: has shape [..., n_ctx, d_model]
        """
        torch.cuda.synchronize()
        start_time = time()
        t = time()
        input_shape = X.shape
        n_ctx = X.shape[-2]
        n_heads = self.n_heads
        d_model = self.d_model
        d_head = self.d_head
        assert input_shape[-1] == d_model
        assert d_model == n_heads * d_head
        def split_heads(x):
            return x.view(x.shape[:-1] + (n_heads, d_head)).transpose(-2, -3).contiguous()
        def merge_heads(x):
            x = x.transpose(-2,-3).contiguous()
            return x.view(x.shape[:-2] + (d_model,))
        self.data["preamble"]["energy"] += 0.0
        self.data["preamble"]["time"] += time() - t

        X = self.layernorm(X)
        Q = self.query_projection(X)
        K = self.key_projection(X)
        V = self.value_projection(X)

        torch.cuda.synchronize()

        t = time()
        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)
        torch.cuda.synchronize()
        self.data["split_heads"]["energy"] += 3.0 * (torch.numel(Q) + torch.numel(K) + torch.numel(V))
        self.data["split_heads"]["time"] += time() - t


        t = time()
        QKT = torch.matmul(Q, K.transpose(-1,-2))
        torch.cuda.synchronize()
        self.data["matmul(Q,K^T)"]["energy"] += 3.0 * (torch.numel(Q) * n_ctx)
        self.data["matmul(Q,K^T)"]["time"] += time() - t

        t = time()
        additive_mask = 1.0-1.0/torch.tril(torch.ones(n_ctx,n_ctx, device=X.device))
        # The expected behavior of the last line is:
        #   additive_mask has shape [n_ctx, n_ctx] and is defined by
        #   additive_mask[i,j] := | 0.0    if i >= j
        #                         | -inf   otherwise
        Z = torch.tril(QKT/math.sqrt(d_head)) + additive_mask
        (Zmax, _) = torch.max(Z,dim=-1,keepdim=True)
        Z = Z - Zmax # as in GPT-2
        EZ = torch.exp(Z)
        sumEZ = torch.sum(EZ,dim=-1,keepdim=True)
        A = EZ/sumEZ
        torch.cuda.synchronize()
        self.data["compute_attention"]["energy"] += 3.0 * 8.0 * (torch.numel(A)) # ?
        self.data["compute_attention"]["time"] += time() - t

        t = time()
        AV = torch.matmul(A,V)
        torch.cuda.synchronize()
        self.data["matmul(A,V)"]["energy"] += 3.0 * (torch.numel(A) * d_head)
        self.data["matmul(A,V)"]["time"] += time() - t

        t = time()
        AV = merge_heads(AV)
        torch.cuda.synchronize()
        self.data["merge_heads"]["energy"] += 3.0 * (torch.numel(AV))
        self.data["merge_heads"]["time"] += time() - t

        Y = self.output_projection(AV)
        torch.cuda.synchronize()
        self.compute_time += time() - start_time
        return Y


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.layernorm = MyLayerNorm(d_model)
        self.layer0 = MyLinear(d_model, d_ff)
        self.nonlinear = torch.nn.GELU()
        self.layer1 = MyLinear(d_ff, d_model)
        self.compute_time = 0.0
        self.energy = 0.0

    def compute_data(self):
        data = {"name": "FeedForward",
                "time": self.compute_time,
                "children": {
                    "layernorm": self.layernorm.compute_data(),
                    "layer0": self.layer0.compute_data(),
                    "nonlinear": {"energy": 0.0}, # TODO
                    "layer1": self.layer1.compute_data()}}
        data["energy"] = sum(data["children"][key]["energy"] for key in data["children"])
        return data

    @autocast()
    def forward(self, x):
        start_time = time()
        x = self.layernorm(x)
        x = self.layer0(x)
        x = self.nonlinear(x)
        x = self.layer1(x)
        torch.cuda.synchronize()
        self.compute_time += time() - start_time
        return x


class TransformerLayer(torch.nn.Module):
    def __init__(self, d_model=64, n_heads=8, d_ff=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.multiheadselfattention = MultiHeadSelfAttention(d_model, n_heads)
        self.feedforward = FeedForward(d_model, d_ff)
        self.compute_time = 0.0

    def compute_data(self):
        data = {"name": "TransformerLayer",
                "time": self.compute_time,
                "children": {
                    "multiheadselfattention": self.multiheadselfattention.compute_data(),
                    "feedforward": self.feedforward.compute_data()}}
        data["energy"] = sum(data["children"][key]["energy"] for key in data["children"])
        return data

    @autocast()
    def forward(self, x):
        start_time = time()
        x = x + self.multiheadselfattention(x)
        x = x + self.feedforward(x)
        torch.cuda.synchronize()
        self.compute_time += time() - start_time
        return x


class MyEmbedding(torch.nn.Module):
    def __init__(self, n_vocab, d_model):
        super().__init__()
        self.n_vocab = n_vocab
        self.d_model = d_model
        self.weight = torch.nn.Parameter(0.01*torch.randn(n_vocab, d_model))
        # TODO: figure out why 1.00 instead of 0.01 causes a nan in gradient
        # for transformer
        self.compute_time = 0.0

    def compute_data(self):
        data = {"name": "MyEmbedding",
                "time": self.compute_time,
                "energy": 0.0}  #TODO
        return data

    def forward(self, x):
        start_time = time()
        shape = x.shape + (self.d_model,)
        result = torch.index_select(self.weight, 0, x.view(-1)).view(shape)
        torch.cuda.synchronize()
        self.compute_time += time() - start_time
        return result


class Transformer(torch.nn.Module):
    """
    Transformer for Generative Language Model; similar to GPT2's design.
    """
    def __init__(self,
                 n_vocab=256,
                 max_ctx=1024,
                 d_model=768,
                 n_heads=12,
                 d_ff=4096,
                 n_layers=12):
        super().__init__()
        self.n_vocab = n_vocab
        self.max_ctx = max_ctx
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.input_embedding = MyEmbedding(n_vocab, d_model)
        self.positional_encoding = torch.nn.Parameter(0.02*torch.randn(max_ctx, d_model))
        self.layers = ModuleList([TransformerLayer(d_model, n_heads, d_ff)
                                  for _ in range(n_layers)])
        self.layernorm = MyLayerNorm(d_model)
        #self.decoder = Linear(d_model, n_vocab)
        self.compute_time = 0.0

    def compute_data(self):
        data = {"name": "Transformer",
                "time": self.compute_time,
                "children": {
                    "input_embedding": self.input_embedding.compute_data(),
                    "positional_encoding": None,
                    "layers": [layer.compute_data() for layer in self.layers],
                    "layernorm": self.layernorm.compute_data()}}
        data["energy"] = (sum(layerdata["energy"] for layerdata in data["children"]["layers"]) +
                          data["children"]["layernorm"]["energy"])
        return data

    def compute_energy(self, L):
        d_model = self.d_model
        n_heads = self.n_heads
        d_ff = self.d_ff
        n_vocab = self.n_vocab
        n_layers = self.n_layers
        d_head = d_model // n_heads
        embedding_cost = d_model * (L-1)
        layernorm_cost = 4.0 * d_model * (L-1) # estimate
        residual_cost = d_model * (L-1)
        attention_cost = 2.0 * (L-1)**2 * d_head * n_heads + layernorm_cost + residual_cost
        feedforward_cost = 2.0 * d_model * d_ff + layernorm_cost + residual_cost
        layer_cost = attention_cost + feedforward_cost
        decoder_cost = d_model * n_vocab * (L-1) + layernorm_cost
        return (3.0*(embedding_cost + n_layers * layer_cost + decoder_cost))/1.0E12

    @autocast()
    def forward(self, X, probs=False):
        """
        input
          X has dtype long. It gives vocab indices.
          Define n_ctx = X.shape[-1] - 1. We subtract one because we use the last
          token only as a classification label.

          probs: if True, then
                 input is considered to have shape [..., n_ctx]
                 output gives an [..., n_ctx+1, n_vocab] matrix of
                 probabilities giving the predictions of the model. See also self.probs
        output
          has shape [..., n_ctx] and contains the loss of its ... * n_ctx predictions.
          (unless probs=True is set, in which case the output is as described above)
        """
        start_time = time()
        if not probs:
            Y = X[...,1:].contiguous()
            X = X[...,:-1].contiguous()
        if torch.any(torch.isnan(X)):
            raise RuntimeError(f"Transformer.forward: Detected nan in input.")
        X = self.input_embedding(X)
        n_ctx = X.shape[-2]
        assert n_ctx <= self.max_ctx
        # Now X has shape [..., n_ctx, d_model] and Y has shape [..., n_ctx]
        if torch.any(torch.isnan(X)):
            raise RuntimeError(f"Transformer.forward: Detected nan after embedding.")
        X = X + self.positional_encoding[...,-n_ctx:,:]
        assert X.shape[-1] == self.d_model
        for (idx, layer) in enumerate(self.layers):
            X = layer(X)
            if torch.any(torch.isnan(X)):
                raise RuntimeError(f"Transformer.forward: Detected nan after layer {idx}.")
        X = self.layernorm(X)
        X = X @ self.input_embedding.weight.transpose(0,1)
        assert X.shape[-1] == self.n_vocab
        if torch.any(torch.isnan(X)):
            raise RuntimeError(f"Transformer.forward: Detected nan after decoding.")

        # Question: How hard would it be to do instead of a transpose a pinv-like calculation?

        # Now X has shape [..., n_ctx, n_vocab] and Y has shape [..., n_ctx]
        if probs:
            # Softmax(dim=-1)
            EX = torch.exp(X)
            sumEX = torch.sum(EX,dim=-1,keepdim=True)
            result = EX/sumEX
        else:
            # Per example, per token crossentropy loss
            EX = torch.exp(X)
            logsumEX = torch.log(torch.sum(EX,dim=-1)).view(-1)
            chosen = torch.index_select(X.view(-1,self.n_vocab),-1,Y.view(-1))
            result = (logsumEX - chosen)/math.log(2)
        if torch.any(torch.isnan(result)):
            raise RuntimeError(f"Transformer.forward: Detected nan after criterion.")
        torch.cuda.synchronize()
        self.compute_time += time() - start_time
        return result

    def probs(self, X):
        with torch.no_grad():
            return self.forward(X, probs=True)

    def name(self):
        return f"Transformer({self.n_vocab},{self.max_ctx},{self.d_model},{self.n_heads},{self.d_ff},{self.n_layers})"
