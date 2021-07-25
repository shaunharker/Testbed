import torch
from torch.nn import Module, Embedding, Linear, CrossEntropyLoss, Softmax
import math
from torch.cuda.amp import autocast

class Net3(Module):
    def __init__(self,
                 num_input_classes=256,
                 embedding_dim=32,
                 context_length=128,
                 num_hidden1=4096,
                 num_hidden2=4096,
                 num_output_classes=256):
        super().__init__()
        self.num_input_classes = num_input_classes
        self.embedding_dim = embedding_dim # dimension of token embedding
        self.context_length = context_length # context window length
        self.num_hidden1 = num_hidden1 # number of hidden neurons
        self.num_hidden2 = num_hidden2
        self.num_output_classes= num_output_classes
        self.embedding = Embedding(self.num_input_classes, self.embedding_dim)
        self.encoder = Linear(self.embedding_dim*self.context_length, self.num_hidden1)
        self.middle = Linear(self.num_hidden1, self.num_hidden2)
        self.decoder = Linear(self.num_hidden2, self.num_output_classes)
        self.criterion = CrossEntropyLoss(reduction='none')
        self.softmax = Softmax(dim=-1)
        self.nonlinear = torch.nn.GELU()

    def name(self):
        return f"Net3({self.embedding_dim},{self.context_length},{self.num_hidden1},{self.num_hidden2})"

    def compute_energy(self, example_length=None):
        return 3.0*(self.embedding_dim*self.context_length*self.num_hidden1 +
                    self.num_hidden1*self.num_hidden2 +
                    self.num_hidden2*self.num_output_classes +
                    self.num_hidden1 +
                    self.num_hidden2 +
                    self.num_output_classes)/1.0E12

    @autocast()
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
        x = self.encoder(x) # x.shape == [..., self.num_hidden1]
        x = self.nonlinear(x) # x.shape == [..., self.num_hidden]
        x = self.middle(x) # x.shape == [..., self.num_hidden2]
        x = self.nonlinear(x) # x.shape == [..., self.num_hidden2]
        x = self.decoder(x) # x.shape == [..., 256]
        loss = self.criterion(x,y)/math.log(2)
        return loss

    def probs(self, X):
        """
        input is tensor of shape 1 x L, long.
        """
        L = self.context_length
        K = self.embedding_dim
        x = self.embedding(X[...,-self.context_length:]) # x.shape == [..., self.context_length, self.embedding_dim]
        x = x.view(-1,self.context_length*self.embedding_dim) # x.shape == [..., self.context_length*self.embedding_dim]
        x = self.encoder(x) # x.shape == [..., self.num_hidden1]
        x = self.nonlinear(x) # x.shape == [..., self.num_hidden1]
        x = self.middle(x) # x.shape == [..., self.num_hidden2]
        x = self.nonlinear(x) # x.shape == [..., self.num_hidden2]
        x = self.decoder(x) # x.shape == [..., 256]
        P = self.softmax(x) # P.shape == [..., 256]
        return P
