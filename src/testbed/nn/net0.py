import torch
from torch.nn import Module, Embedding, Linear, CrossEntropyLoss, Softmax
import math
from torch.cuda.amp import autocast
from .poly import Poly

class Net0(Module):
    def __init__(self,
                 num_input_classes=256,
                 embedding_dim=32,
                 context_length=32,
                 num_hidden=8192,
                 num_output_classes=256,
                 nonlinearity='sigmoid',
                 degree=None):
        super().__init__()
        self.num_input_classes = num_input_classes # number of classes
        self.embedding_dim = embedding_dim # dimension of token embedding
        self.context_length = context_length # context window length
        self.num_hidden = num_hidden # number of hidden neurons
        self.num_output_classes = num_output_classes
        self.embedding = Embedding(self.num_input_classes, self.embedding_dim)
        self.encoder = Linear(self.embedding_dim*self.context_length, self.num_hidden)
        self.nonlinearity = nonlinearity
        if nonlinearity == 'sigmoid':
            self.nonlinear = torch.nn.Sigmoid()
        if nonlinearity == 'GELU':
            self.nonlinear = torch.nn.GELU()
        if nonlinearity == 'sin':
            self.nonlinear = torch.sin
        if nonlinearity == 'poly':
            if degree is None:
                degree = 7
            self.degree = degree
            self.nonlinear = Poly(degree=self.degree)
        self.decoder = Linear(self.num_hidden, self.num_output_classes)
        self.criterion = CrossEntropyLoss(reduction='none')
        self.softmax = Softmax(dim=-1)

    def name(self):
        return f"Net0({self.num_input_classes},{self.embedding_dim},{self.context_length},{self.num_hidden},{self.num_output_classes},{self.nonlinearity})"

    def compute_energy(self, example_length=None):
        return 3.0*(self.embedding_dim*self.context_length*self.num_hidden + self.num_hidden*self.num_output_classes + self.num_hidden + self.num_output_classes)/1.0E12

#    @autocast()
    def forward(self, batch):
        """
        example_length = batch.shape[-1]
        assert example_length = self.context_length + 1
        """
        example_length = batch.shape[-1]
        assert example_length == self.context_length + 1
        x = batch[...,:-1] # x.shape
        x = self.embedding(x) # x.shape == [..., self.context_length, self.embedding_dim]
        y = batch[...,-1].view(-1) # y.shape == [...]
        x = x.view(-1,self.context_length*self.embedding_dim) # x.shape == [..., self.context_length*self.embedding_dim]
        x = self.encoder(x) # x.shape == [..., self.num_hidden]
        x = self.nonlinear(x) # x.shape == [..., self.num_hidden]
        x = self.decoder(x) # x.shape == [..., self.num_output_classes]
        loss = self.criterion(x,y)/math.log(2)
        return loss

    def probs(self, X):
        """
        input is tensor of shape 1 x L, long.
        """
        L = self.context_length
        K = self.embedding_dim
        x = self.embedding(X[...,-self.context_length:]) # x.shape == [B, L, K]
        x = x.view(-1,self.context_length*self.embedding_dim) # x.shape == [..., self.context_length*self.embedding_dim]
        x = self.encoder(x) # x.shape == [..., self.num_hidden]
        x = self.nonlinear(x) # x.shape == [..., self.num_hidden]
        x = self.decoder(x) # x.shape == [..., self.num_output_classes]
        P = self.softmax(x) # P.shape == [..., self.num_output_classes]
        return P
