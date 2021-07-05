import torch
from torch.nn import Module, Embedding, Linear, CrossEntropyLoss, Softmax
import math
from torch.nn.functional import pad
import numpy as np


class MLP(Module):
    def __init__(self, num_input, num_hidden, num_output):
        super().__init__()
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.encoder = Linear(self.num_input, self.num_hidden)
        self.decoder = Linear(self.num_hidden, self.num_output)

    def size(self):
        return sum(torch.numel(p) for p in self.parameters())

    def forward(self, x):
        return self.decoder(torch.sigmoid(self.encoder(x)))


class Net4(Module):
    def __init__(self,
                 Cin=256,
                 E=16,
                 L=64,
                 M=64,
                 H=64,
                 Cout=256,
                 context_length=64):
        super().__init__()
        self.E = E
        self.L = L
        self.M = M
        self.H = H
        self.Cin = Cin
        self.Cout = Cout
        self.embedding = Embedding(Cin, E)
        # complex encoder
        self.W = MLP(M + L*E, H, M*M)
        self.b = MLP(M + L*E, H, M)
        # decoder
        self.decoder = Linear(M, Cout)
        self.criterion = CrossEntropyLoss(reduction='none')
        self.softmax = Softmax(dim=-1)

    def size(self):
        return sum(torch.numel(p) for p in self.parameters())

    def name(self):
        return f"Net4({self.E},{self.L},{self.M},{self.H},size={self.size()})"

    def forward(self, x0):
        """
        x : [..., seq_length]
        """
        prefix = x0.shape[:-1]
        N = x0.shape[-1]
        L = self.L
        E = self.E
        M = self.M
        m = torch.zeros(prefix + (M,), device=x0.device)
        losses = []
        for idx in range(N):
            offset = idx-L
            x = x0[...,max(0,offset):idx]
            #print(f"Line 69. x.shape {x.shape}, x0.shape {x0.shape}, offset {offset}, idx {idx}")
            if offset < 0:
                #print(f"Line 71. Padding by {-offset}. idx = {idx}. N = {N}.")
                x = pad(x,(-offset, 0))
            y = x0[...,idx]
            #print(f"Line 74. x.shape {x.shape} prefix {prefix} L {L} E {E} L*E {L*E}")
            x = self.embedding(x)
            #print(f"Line 76. x.shape {x.shape}")
            x = x.view(*prefix, L*E)
            #print(f"Line 78. x.shape {x.shape} m.shape {m.shape}")
            x = torch.cat([m,x],dim=-1)
            #print(f"Line 80. x.shape {x.shape}")
            W = self.W(x).view(*prefix,M,M)
            b = self.b(x).view(*prefix,M)
            #print(f"Line 83. W.shape {W.shape}, m.shape {m.shape}, b.shape {b.shape}")
            m = torch.sigmoid(torch.matmul(W,m.view(m.shape + (1,))).view(b.shape) + b)
            x = self.decoder(m)
            losses.append(self.criterion(x,y)/math.log(2))
        return torch.cat(losses)

    @torch.no_grad()
    def probs(self, bytes_object):
        """
        Given a bytes object, read the entirety of it with this RNN,
        and then provide the matrix of next-byte predictions as
        an [N, 256] numpy array, so that P[idx] gives the prediction for
        the idx+1 byte in the bytes_object (the last one not actually being there)
        """
        # We add the 0 to get it to predict the last char
        x0 = torch.tensor(list(bytes_object),device=self.decoder.weight.device).long()
        N = len(bytes_object)
        L = self.L
        E = self.E
        M = self.M
        m = torch.zeros(M, device=self.decoder.weight.device)
        P = []
        for idx in range(N+1):
            offset = idx-L
            x = x0[max(0,offset):idx]
            if offset < 0:
                x = pad(x,(-offset, 0))
            x = self.embedding(x).view(L*E)
            x = torch.cat([m,x],dim=-1)
            W = self.W(x).view(M,M)
            b = self.b(x).view(M)
            m = torch.sigmoid(W@m + b)
            x = self.decoder(m)
            P.append(self.softmax(x).view(1,-1).cpu().numpy())
        return np.concatenate(P)
