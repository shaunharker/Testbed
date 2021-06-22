import torch
from torch.nn import Module, Embedding, Conv1d, Linear, Softmax, CrossEntropyLoss

class Net1(Module):
    def __init__(self, H, L, K=8, C=256):
        super(Net1, self).__init__()
        self.C = C # number of classes
        self.K = K # dimension of token embedding
        self.L = L # context window length
        self.H = H # number of hidden neurons
        self.embedding = Embedding(C, K)
        self.layer0 = Conv1d(K, H, L)
        self.layer1 = Linear(H, C)
        self.softmax = Softmax(dim=-1)
        self.criterion = CrossEntropyLoss()
        self.batch_first = True

    def name(self):
        return f"net1_H{self.H}_L{self.L}_K{self.K}_C{self.C}"

    def forward(self, X):
        """
        Input:
        X is a Tensor with shape [N, B] holding integers in {0...K-1}
        We understand this as columns of contiguous text where {0...K-1} is
        the alphabet. N > L required.

        N is batch example length.
        B is number of examples in a batch (i.e. batch size)
        K is embedding dimension
        C is number of classifications ()
        """
        x = self.embedding(X[:,:-1]) # x.shape == [B, N-1, K]
        y = X[:,self.L:] # y.shape == [B, N-L]
        x = x.transpose(1,2) # x.shape == [B, K, N-1]
        x = self.layer0(x) # x.shape == [B, H, N-L]
        x = x.transpose(1,2) # x.shape == [B, N-L, H]
        x = torch.sigmoid(x) # x.shape == [B, N-L, H]
        x = self.layer1(x) # x.shape == [B, N-L, C]
        x = x.view(-1, self.C) # x.shape = [B*(N-L), C]
        y = y.reshape(-1) # y.shape == [B*(N-L)]
        return self.criterion(x, y)

    def probs(self, X): # X.shape == [B, N]
        """
        N > L is required
        """
        x = self.embedding(X) # x.shape == [B, N, K]
        x = x.transpose(1,2) # x.shape == [B, K, N]
        x = self.layer0(x) # x.shape == [B, H, N-L+1]
        x = x.transpose(1,2) # x.shape == [B, N-L+1, H]
        x = torch.sigmoid(x) # x.shape == [B, N-L+1, H]
        x = self.layer1(x) # x.shape == [B, N-L+1, C]
        P = self.softmax(x)[:,-1] # P.shape == [B, C]
        return P
