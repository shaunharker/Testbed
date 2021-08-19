import math
import torch
from torch.nn import Module, ModuleList, Sigmoid, ReLU, GELU, LayerNorm
from torch.nn import Embedding as TorchEmbedding
from torch.nn import Linear as TorchAffine
import dill
from types import GeneratorType
import copy

class SplitExample(Module):
    def __init__(self, mode="last"):
        super().__init__()
        self.mode = mode

    def forward(self, xy):
        if self.mode == "last":
            return (xy[...,:-1].contiguous(), xy[...,-1].contiguous())
        elif self.mode == "shift":
            n = xy.shape[-1]
            return (xy[...,:-1].contiguous(), xy[...,1:].contiguous())


class Embedding(TorchEmbedding):
    def __init__(self, n_classes, d_model):
        super().__init__(n_classes, d_model)
        self.n_classes = n_classes
        self.d_model = d_model


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        layers = sum([list(layer) if type(layer)==GeneratorType else [layer] for layer in layers],[])
        self.layers = ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Lambda(Module):
    def __init__(self, F):
        super().__init__()
        self.F = F

    def forward(self, x):
        return self.F(x)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['F'] = dill.dumps(self.F)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.F = dill.loads(self.F)


class Affine(TorchAffine):
    def __init__(self, d_in, d_out, bias=True):
        super().__init__(d_in, d_out, bias)
        self.d_in = d_in
        self.d_out = d_out


class Nonlinearity(Module):
    def __init__(self, nonlinearity):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.F = {"sigmoid": Sigmoid(), "ReLU": ReLU(), "GELU": GELU()}[nonlinearity]

    def forward(self, x):
        return self.F(x)


class CrossEntropyLoss(Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.F = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, y):
        return self.F(x.reshape(-1,self.n_classes), y.reshape(-1)).view(x.shape[:-1])/math.log(self.n_classes)


class Softmax(Module):
    def __init__(self):
        super().__init__()
        self.F = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        return self.F(x)


class MLP(Module):
    def __init__(self, d_in, d_hidden, nonlinearity, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.nonlinearity = nonlinearity
        self.F = Sequential(Affine(d_in=d_in, d_out=d_hidden, bias=False), LayerNorm(d_hidden), Nonlinearity(nonlinearity), Affine(d_in=d_hidden, d_out=d_out))

    def forward(self, x):
        return self.F(x)

    @torch.no_grad()
    def canonicalize(self):
        W = self.F.layers[0].weight
        N = torch.numel(W)
        W -= torch.mean(W)
        W /= math.sqrt(torch.sum(W*W)/(N-1))
        W.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)


class LanguageModel(Module):
    def __init__(self, F, n_vocab_out, mode):
        super().__init__()
        self.F = F
        self.split_example = SplitExample(mode)
        self.crossentropyloss = CrossEntropyLoss(n_vocab_out)
        self.softmax = Softmax()

    def forward(self, xy):
        (x, y) = self.split_example(xy)
        x = self.F(x)
        return self.crossentropyloss(x, y)/x.shape[-2]

    @torch.no_grad()
    def inference(self, x):
        return self.softmax(self.F(x))

class LanguageModel2(Module):
    def __init__(self, F, n_vocab_out, mode):
        super().__init__()
        self.F = F
        self.n_vocab_out = n_vocab_out
        self.mode = mode
        self.split_example = SplitExample(mode)

    def forward(self, xy):
        (x, y) = self.split_example(xy)
        n_ctx = x.shape[-2]
        x = self.F(x)
        x = x**2 # torch.exp(x)
        x = x / torch.sum(x,dim=-1,keepdim=True)
        p = 1/self.n_vocab_out
        x = torch.nan_to_num(x, nan=p, posinf=p, neginf=p)
        result = torch.gather(input=x.view(-1,self.n_vocab_out), dim=-1, index=y.view(-1, 1))
        #print(f"A {result.shape} torch.gather(input=x.view(-1,self.n_vocab_out), dim=-1, index=y.view(-1, 1))={torch.gather(input=x.view(-1,self.n_vocab_out), dim=-1, index=y.view(-1, 1))}")
        #print(f"B x[0]={x[0]} with shape {x[0].shape} (x has shape {x.shape}) sum = {torch.sum(x[0])}")
        result = -torch.log(result)/math.log(self.n_vocab_out)
        result = torch.clamp(torch.nan_to_num(result),min=0.0,max=2.0)
        #print(f"C {result}, mean = {torch.mean(result)}, result shape = {result.shape}")
        #raise hell
        return result

    @torch.no_grad()
    def inference(self, x):
        x = self.F(x)
        x = x**2 # torch.exp(x)
        x = x / torch.sum(x)
        return x

class MLPLM(Module):
    def __init__(self, n_ctx, n_vocab_in, d_model, d_hidden, nonlinearity, n_vocab_out):
        super().__init__()
        self.n_ctx = n_ctx
        self.n_vocab_in = n_vocab_in
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.nonlinearity = nonlinearity
        self.n_vocab_out = n_vocab_out
        self.LM = LanguageModel(Sequential(Embedding(n_classes=n_vocab_in, d_model=d_model), Lambda(lambda x: x.view(-1,n_ctx*d_model)), MLP(d_in=n_ctx*d_model, d_hidden=d_hidden, nonlinearity=nonlinearity, d_out=n_vocab_out), Lambda(lambda x: x.view(-1, 1, n_vocab_out))), n_vocab_out=n_vocab_out, mode="last")

    def forward(self, x):
        return self.LM(x)

    @torch.no_grad()
    def inference(self, x):
        return self.LM.inference(x)

    def clone(self):
        return copy.deepcopy(self)

    @torch.no_grad()
    def canonicalize(self):
        E = self.LM.F.layers[0].weight
        N = torch.numel(E)
        Emu = torch.mean(E)
        E -= Emu
        E /= math.sqrt(torch.sum(E**2/(N-1)))
        E.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        MLP = self.LM.F.layers[2]
        W = MLP.F.layers[0].weight # affine weight
        G = MLP.F.layers[1].weight # layernorm gain
        B = MLP.F.layers[1].bias # layernorm bias
        B += torch.sum(W,dim=1).view(-1)*G*Emu
        MLP.canonicalize()

class MLPLM2(Module):
    def __init__(self, n_ctx, n_vocab_in, d_model, d_hidden, nonlinearity, n_vocab_out):
        super().__init__()
        self.n_ctx = n_ctx
        self.n_vocab_in = n_vocab_in
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.nonlinearity = nonlinearity
        self.n_vocab_out = n_vocab_out
        self.LM = LanguageModel2(Sequential(Embedding(n_classes=n_vocab_in, d_model=d_model), Lambda(lambda x: x.view(-1,n_ctx*d_model)), MLP(d_in=n_ctx*d_model, d_hidden=d_hidden, nonlinearity=nonlinearity, d_out=n_vocab_out), Lambda(lambda x: x.view(-1, 1, n_vocab_out))), n_vocab_out=n_vocab_out, mode="last")

    def forward(self, x):
        return self.LM(x)

    @torch.no_grad()
    def inference(self, x):
        return self.LM.inference(x)

    def clone(self):
        return copy.deepcopy(self)

    @torch.no_grad()
    def canonicalize(self):
        E = self.LM.F.layers[0].weight
        N = torch.numel(E)
        Emu = torch.mean(E)
        E -= Emu
        E /= math.sqrt(torch.sum(E**2/(N-1)))
        E.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        MLP = self.LM.F.layers[2]
        W = MLP.F.layers[0].weight # affine weight
        G = MLP.F.layers[1].weight # layernorm gain
        B = MLP.F.layers[1].bias # layernorm bias
        B += torch.sum(W,dim=1).view(-1)*G*Emu
        MLP.canonicalize()
        W = MLP.F.layers[3].weight # output weight
        B = MLP.F.layers[3].bias # output bias
        s = math.sqrt((torch.sum(W**2) + torch.sum(B**2)) / (torch.numel(W) + torch.numel(B) - 1))
        W /= s
        B /= s
