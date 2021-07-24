import math
import torch
import torch.nn.functional as F
import time
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Dropout, Embedding, Linear, CrossEntropyLoss, Softmax, LayerNorm, ModuleList


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        d_head = d_k = d_v = d_model // n_heads # assume these are equal for this implementation
        self.d_head = d_head
        self.layernorm = LayerNorm(d_model)
        self.query_projection = Linear(d_model, d_k * n_heads)
        self.key_projection = Linear(d_model, d_k * n_heads)
        self.value_projection = Linear(d_model, d_v * n_heads)
        self.attention_softmax = torch.nn.Softmax(dim=-1) # Note: in contrast GPT2 implementation at https://github.com/openai/gpt-2/blob/master/src/model.py shifts values to make the max entry 0 before they do softmax
        self.output_projection = Linear(d_v * n_heads, d_model)

    def forward(self, X):
        """
        input: X has shape [..., n_ctx, d_model]
        output: has shape [..., n_ctx, d_model]
        """
        input_shape = X.shape
        n_ctx = X.shape[-2]
        n_heads = self.n_heads
        d_model = self.d_model
        d_head = self.d_head
        assert input_shape[-1] == d_model
        assert d_model == n_heads * d_head
        X = self.layernorm(X)
        Q = self.query_projection(X)
        K = self.key_projection(X)
        V = self.value_projection(X)
        #print('1 X', X.shape)
        #print('1 Q', Q.shape)
        #print('1 K', K.shape)
        #print('1 V', V.shape)
        def split_heads(x):
            #print('split_heads', x.shape)
            return x.view(x.shape[:-1] + (n_heads, d_head)).transpose(-2, -3).contiguous()
        def merge_heads(x):
            #print('1 merge_heads', x.shape, d_model)
            x = x.transpose(-2,-3).contiguous()
            #print('2 merge_heads', x.shape, d_model)
            return x.view(x.shape[:-2] + (d_model,))
        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        #print('2 Q', Q.shape)
        #print('2 K', K.shape)
        #print('2 V', V.shape)
        QKT = torch.matmul(Q, K.transpose(-1,-2))

        additive_mask = 1.0-1.0/torch.tril(torch.ones(n_ctx,n_ctx, device=X.device))
        # The expected behavior of the last line is:
        #   additive_mask has shape [n_ctx, n_ctx] and is defined by
        #   additive_mask[i,j] := | 0.0    if i >= j
        #                         | -inf   otherwise

        A = self.attention_softmax(torch.tril(QKT/math.sqrt(d_head))+additive_mask)
        AV = merge_heads(torch.matmul(A,V))
        Y = self.output_projection(AV)
        return Y


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.layernorm = LayerNorm(d_model)
        self.layer0 = Linear(d_model, d_ff)
        self.nonlinear = torch.nn.GELU()
        self.layer1 = Linear(d_ff, d_model)

    def forward(self, x):
        x = self.layernorm(x)
        x = self.layer0(x)
        x = self.nonlinear(x)
        x = self.layer1(x)
        return x


class TransformerLayer(torch.nn.Module):
    def __init__(self, d_model=64, n_heads=8, d_ff=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.multiheadselfattention = MultiHeadSelfAttention(d_model, n_heads)
        self.feedforward = FeedForward(d_model, d_ff)

    def forward(self, x):
        x = x + self.multiheadselfattention(x)
        x = x + self.feedforward(x)
        return x


class Transformer(torch.nn.Module):
    """
    Transformer for Generative Language Model; similar to GPT2's design.
    """
    def __init__(self,
                 n_vocab=256,
                 max_ctx=512,
                 d_model=64,
                 n_heads=8,
                 d_ff=2048,
                 n_layers=6):
        super().__init__()
        self.n_vocab = n_vocab
        self.max_ctx = max_ctx
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.input_embedding = Embedding(n_vocab, d_model)
        self.positional_encoding = torch.nn.Parameter(0.02*torch.randn(max_ctx, d_model))
        self.layers = ModuleList([TransformerLayer(d_model, n_heads, d_ff)
                                  for _ in range(n_layers)])
        self.decoder = Linear(d_model, n_vocab)

    def forward(self, X, probs=False):
        """
        input
          X has shape [..., n_ctx+1] and dtype long. It gives vocab indices.
          probs: if True, then output gives an [..., n_ctx+1, n_vocab] matrix of
                 probabilities giving the predictions of the model. See also self.probs
        output
          has shape [..., n_ctx] and contains the loss of its ... * n_ctx predictions.
          (unless probs=True is set, in which case the output is as described above)
        """
        Y = X[...,1:].contiguous()
        X = X[...,:-1].contiguous()
        X = self.input_embedding(X)
        n_ctx = X.shape[-2]
        assert n_ctx <= self.max_ctx
        # Now X has shape [..., n_ctx, d_model] and Y has shape [..., n_ctx]
        X = X + self.positional_encoding[...,-n_ctx:,:]
        for layer in self.layers:
            X = layer(X)
        X = self.decoder(X)
        # Now X has shape [..., n_ctx, n_vocab] and Y has shape [..., n_ctx]
        if probs:
            # Softmax(dim=-1)
            EX = torch.exp(X)
            sumEX = torch.sum(EX,dim=-1,keepdim=True)
            return EX/sumEX
        else:
            # Per example, per token crossentropy loss
            EX = torch.exp(X)
            logsumEX = torch.log(torch.sum(torch.exp(X),dim=-1)).view(-1)
            chosen = torch.index_select(X.view(-1,self.n_vocab),-1,Y.view(-1))
            #print('crossentropyloss', logsumEX.shape, chosen.shape)
            return (logsumEX - chosen)/math.log(2)

    def probs(self, X):
        with torch.no_grad():
            return self.forward(X, probs=True)

    def name(self):
        return f"Transformer({self.n_vocab},{self.max_ctx},{self.d_model},{self.n_heads},{self.d_ff},{self.n_layers})"
