import torch
from torch.nn import Module, Embedding, Linear, CrossEntropyLoss, Softmax

class Net0(Module):
    def __init__(self, H=256, L=64, K=8, C=256):
        super(Net0, self).__init__()
        self.C = C # number of classes
        self.K = K # dimension of token embedding
        self.L = L # context window length
        self.H = H # number of hidden neurons
        self.embedding = Embedding(C, K)
        self.fc1 = Linear(L*K, H)
        self.fc2 = Linear(H, C)
        self.criterion = CrossEntropyLoss()
        self.softmax = Softmax(dim=-1)

    def name(self):
        return f"net0_H{self.H}_L{self.L}_K{self.K}_C{self.C}"

    def forward(self, X):
        """
        Requires N = L + 1, where X.shape == [N, B]
        """
        L = self.L
        K = self.K
        x = self.embedding(X[:,-L-1:-1]) # x.shape == [B, L, K]
        y = X[:,-1].view(-1) # y.shape == [B]
        x = x.view(-1,L*K)  # s.shape == [B, L*K]
        x = self.fc2(torch.sigmoid(self.fc1(x))) # x.shape == [B, C]
        loss = self.criterion(x,y)
        return loss

    def probs(self, X):
        L = self.L
        K = self.K
        x = self.embedding(X[:,-L:]) # x.shape == [B, L, K]
        x = x.view(-1,L*K)  # s.shape == [B, L*K]
        x = self.fc2(torch.sigmoid(self.fc1(x))) # x.shape == [B, C]
        P = self.softmax(x)
        return P
