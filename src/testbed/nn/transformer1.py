import math
from functools import lru_cache
from time import time
import torch
from torch.nn import Dropout, Embedding, Linear, CrossEntropyLoss, Softmax, LayerNorm, ModuleList
from torch.cuda.amp import autocast
from .reference import MyEmbedding, MyLinear, MyLayerNorm, MyGELU


@lru_cache
def additive_lt_mask(n_ctx, device):
    """
    #   additive_lt_mask returns a torch.Tensor with shape [n_ctx, n_ctx]
    #   defined by:
    #
    #     additive_mask[i,j] := | 0.0    if i >= j
    #                           | -inf   otherwise
    """
    return 1.0-1.0/torch.tril(torch.ones(n_ctx,n_ctx, device=device))

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, p_dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        d_head = d_k = d_v = d_model // n_heads # assume these are equal for this implementation
        self.d_head = d_head
        self.layernorm = MyLayerNorm(d_model)
        self.query_projection = MyLinear(d_model, d_k * n_heads)
        self.key_projection = MyLinear(d_model, d_k * n_heads)
        self.value_projection = MyLinear(d_model, d_v * n_heads)
        self.output_projection = MyLinear(d_v * n_heads, d_model)
        self.dropout = Dropout(p_dropout)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.info = {
            "name": "MultiHeadSelfAttention",
            "time": 0.0,
            "energy": 0.0,
            "children": {
                "layernorm": {"energy": 0.0, "time": 0.0},
                "query_projection": {"energy": 0.0, "time": 0.0},
                "key_projection": {"energy": 0.0, "time": 0.0},
                "value_projection": {"energy": 0.0, "time": 0.0},
                "split_heads": {"energy": 0.0, "time": 0.0},
                "matmul(Q,K^T)": {"energy": 0.0, "time": 0.0},
                "dropout_attention": {"energy": 0.0, "time": 0.0},
                "masking_attention": {"energy": 0.0, "time": 0.0},
                "compute_attention": {"energy": 0.0, "time": 0.0},
                "matmul(A,V)": {"energy": 0.0, "time": 0.0},
                "merge_heads": {"energy": 0.0, "time": 0.0},
                "output_projection": {"energy": 0.0, "time": 0.0}}}

    def profile(self):
        self.info["children"].update({
            "layernorm": self.layernorm.profile(),
            "query_projection": self.query_projection.profile(),
            "key_projection": self.key_projection.profile(),
            "value_projection": self.value_projection.profile(),
            "output_projection": self.output_projection.profile()})

        self.info["energy"] = sum(self.info["children"][key]["energy"]
                                  for key in self.info["children"])
        return self.info

    def forward(self, X):
        """
        input: X has shape [..., n_ctx, d_model]
        output: has shape [..., n_ctx, d_model]
        """
        torch.cuda.synchronize()
        start_time = time()
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
        self.info["children"]["split_heads"]["energy"] += torch.numel(Q) + torch.numel(K) + torch.numel(V)
        self.info["children"]["split_heads"]["time"] += time() - t

        t = time()
        QKT = torch.matmul(Q/math.sqrt(d_head), K.transpose(-1,-2))
        torch.cuda.synchronize()
        self.info["children"]["matmul(Q,K^T)"]["energy"] += torch.numel(Q) * n_ctx
        self.info["children"]["matmul(Q,K^T)"]["time"] += time() - t

        t = time()
        QKT = self.dropout(QKT)
        torch.cuda.synchronize()
        self.info["children"]["dropout_attention"]["energy"] += 16.0 * (torch.numel(QKT)) # ?
        self.info["children"]["dropout_attention"]["time"] += time() - t

        t = time()
        Z = QKT + additive_lt_mask(n_ctx, device=QKT.device)
        torch.cuda.synchronize()
        self.info["children"]["masking_attention"]["energy"] += 4.0 * (torch.numel(Z)) # ?
        self.info["children"]["masking_attention"]["time"] += time() - t

        t = time()
        A = self.softmax(Z)
        torch.cuda.synchronize()
        self.info["children"]["compute_attention"]["energy"] += 15.0 * (torch.numel(A)) # ?
        self.info["children"]["compute_attention"]["time"] += time() - t

        t = time()
        AV = torch.matmul(A,V)
        torch.cuda.synchronize()
        self.info["children"]["matmul(A,V)"]["energy"] += torch.numel(A) * d_head
        self.info["children"]["matmul(A,V)"]["time"] += time() - t

        t = time()
        mergedAV = merge_heads(AV)
        torch.cuda.synchronize()
        self.info["children"]["merge_heads"]["energy"] += torch.numel(AV)
        self.info["children"]["merge_heads"]["time"] += time() - t

        Y = self.output_projection(mergedAV)
        torch.cuda.synchronize()
        self.info["time"] += time() - start_time
        return Y


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, p_dropout):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.layernorm = MyLayerNorm(d_model)
        self.layer0 = MyLinear(d_model, d_ff)
        self.nonlinear = MyGELU(d_ff)
        self.dropout = Dropout(p_dropout)
        self.layer1 = MyLinear(d_ff, d_model)
        self.info = {
            "name": "FeedForward",
            "time": 0.0,
            "energy": 0.0,
            "children": {
                "layernorm": {},
                "layer0": {},
                "nonlinear": {},
                "layer1": {}}}

    def profile(self):
        self.info["children"].update({
            "layernorm": self.layernorm.profile(),
            "layer0": self.layer0.profile(),
            "nonlinear": self.nonlinear.profile(),
            "layer1": self.layer1.profile()})
        self.info["energy"] = sum(self.info["children"][key]["energy"]
                                  for key in self.info["children"])
        return self.info

    def forward(self, x):
        start_time = time()
        x = self.layernorm(x)
        x = self.layer0(x)
        x = self.nonlinear(x)
        x = self.dropout(x)
        x = self.layer1(x)
        torch.cuda.synchronize()
        self.info["time"] += time() - start_time
        return x


class TransformerLayer(torch.nn.Module):
    def __init__(self, d_model=64, n_heads=8, d_ff=2048, p_dropout_attn=0.1, p_dropout_ff=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.p_dropout_attn = p_dropout_attn
        self.p_dropout_ff = p_dropout_ff
        self.multiheadselfattention = MultiHeadSelfAttention(d_model, n_heads, p_dropout_attn)
        self.feedforward = FeedForward(d_model, d_ff, p_dropout_ff)
        self.info = {
            "name": "TransformerLayer",
            "time": 0.0,
            "energy": 0.0,
            "children": {
                "multiheadselfattention": {},
                "feedforward": {}}}

    def profile(self):
        self.info["children"].update({
            "multiheadselfattention": self.multiheadselfattention.profile(),
            "feedforward": self.feedforward.profile()})
        self.info["energy"] = sum(self.info["children"][key]["energy"]
                                  for key in self.info["children"])
        return self.info

    def forward(self, x):
        start_time = time()
        x = x + self.multiheadselfattention(x)
        x = x + self.feedforward(x)
        torch.cuda.synchronize()
        self.info["time"] += time() - start_time
        return x


class Transformer(torch.nn.Module):
    """
    Transformer for Generative Language Model
    """
    def __init__(self,
                 n_vocab=256,
                 max_ctx=1024,
                 n_lm_ctx=32,
                 d_model=768,
                 n_heads=12,
                 d_ff=4096,
                 n_layers=12,
                 p_dropout_in=0.1,
                 p_dropout_attn=0.1,
                 p_dropout_ff=0.1,
                 p_dropout_out=0.1,
                 use_amp=False):
        super().__init__()
        self.n_vocab = n_vocab
        self.max_ctx = max_ctx
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.p_dropout_in = p_dropout_in
        self.p_dropout_attn = p_dropout_attn
        self.p_dropout_ff = p_dropout_ff
        self.p_dropout_out = p_dropout_out

        self.input_embedding = MyEmbedding(n_vocab, d_model)
        self.dropout_in = Dropout(p_dropout_in)
        self.positional_encoding = torch.nn.Parameter(0.02*torch.randn(max_ctx, d_model))
        self.layers = ModuleList([TransformerLayer(d_model,n_heads,d_ff,p_dropout_attn,p_dropout_ff) for _ in range(n_layers)])
        self.layernorm = MyLayerNorm(d_model)
        self.dropout_out = Dropout(p_dropout_out)
        self.lm_head = MyLinear(d_model, n_vocab)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.criterion = CrossEntropyLoss(reduction='none')
        self.info = {
            "name": "Transformer",
            "time": 0.0,
            "energy": 0.0,
            "children": {
                "input_embedding": {},
                "layers": [],
                "layernorm": {}}}

    def profile(self):
        self.info["children"].update({
            "input_embedding": self.input_embedding.profile(),
            "positional_encoding": None,
            "layers": [layer.profile() for layer in self.layers],
            "layernorm": self.layernorm.profile(),
            "lm_head": self.lm_head.profile()})
        self.info["energy"] = (sum(info["energy"] for info in self.info["children"]["layers"]) + self.info["children"]["layernorm"]["energy"])
        return self.info

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
        with autocast(enable=self.use_amp):
            start_time = time()
            if not probs:
                Y = X[...,1:].contiguous()
                X = X[...,:-1].contiguous()
            X = self.input_embedding(X)
            X = self.dropout_in(X)
            n_ctx = X.shape[-2]
            assert n_ctx <= self.max_ctx
            # Now X has shape [..., n_ctx, d_model] and Y has shape [..., n_ctx]
            X = X + self.positional_encoding[...,-n_ctx:,:]
            assert X.shape[-1] == self.d_model
            for (idx, layer) in enumerate(self.layers):
                X = layer(X)
            X = self.layernorm(X)
            X = self.dropout_out(X)
            X = self.lm_head(X)
            assert X.shape[-1] == self.n_vocab
            # Now X has shape [..., n_ctx, n_vocab] and Y has shape [..., n_ctx]
            if probs:
                result = self.softmax(X)
            else:
                result = self.criterion(X[...,n_ctx//2:,:].view(-1,self.n_vocab),Y[...,n_ctx//2:].view(-1)).view(X.shape[:-2]+(-1,))/math.log(self.n_vocab)
            torch.cuda.synchronize()
            self.info["time"] += time() - start_time
            return result

    def probs(self, X):
        with torch.no_grad():
            return self.forward(X, probs=True)

    def name(self):
        return f"Transformer({self.n_vocab},{self.max_ctx},{self.d_model},{self.n_heads},{self.d_ff},{self.n_layers})"
