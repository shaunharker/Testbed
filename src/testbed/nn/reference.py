import torch
from torch.cuda.amp import autocast
from time import time
from torch.nn import Dropout, Embedding, Linear, CrossEntropyLoss, Softmax, LayerNorm, ModuleList

class MyEmbedding(torch.nn.Module):
    def __init__(self, n_vocab, d_model):
        super().__init__()
        self.n_vocab = n_vocab
        self.d_model = d_model
        self.weight = torch.nn.Parameter(0.01*torch.randn(n_vocab, d_model))
        # TODO: figure out why 1.00 instead of 0.01 causes a nan in gradient
        # for transformer
        self.info = {
            "name": "MyEmbedding",
            "time": 0.0,
            "energy": 0.0}

    def profile(self):
        return self.info

    def forward(self, x):
        start_time = time()
        shape = x.shape + (self.d_model,)
        result = torch.index_select(self.weight, 0, x.view(-1)).view(shape)
        torch.cuda.synchronize()
        self.info["energy"] += torch.numel(result)
        self.info["time"] += time() - start_time
        return result


class MyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features)*.02)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.info = {
            "name": "MyLinear",
            "time": 0.0,
            "energy": 0.0}

    def profile(self):
        return self.info

    def forward(self, x):
        start_time = time()
        y = x @ self.weight + self.bias
        torch.cuda.synchronize()
        self.info["energy"] += self.out_features * torch.numel(x) + torch.numel(y)
        self.info["time"] += time() - start_time
        return y


class MyGELU(torch.nn.Module):
    def __init__(self, d_ff):
        super().__init__()
        self.d_ff = d_ff
        self.gelu = torch.nn.GELU()
        self.info = {
            "name": "MyGELU",
            "time": 0.0,
            "energy": 0.0,
            "data": 0.0}

    def profile(self):
        return self.info

    def forward(self, x):
        start_time = time()
        x = self.gelu(x)
        torch.cuda.synchronize()
        self.info["time"] += time() - start_time
        self.info["energy"] += 6.0 * torch.numel(x)
        self.info["data"] += torch.numel(x)
        return x


class MyLayerNorm(LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, eps=1e-4, **kwargs)
        self.info = {
            "name": "MyLayerNorm",
            "time": 0.0,
            "energy": 0.0}

    def profile(self):
        return self.info

    def forward(self, x):
        start_time = time()
        result = super().forward(x)
        torch.cuda.synchronize()
        self.info["energy"] += 6.0 * torch.numel(x) # ?
        self.info["time"] += time() - start_time
        return result
