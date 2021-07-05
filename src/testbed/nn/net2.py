import torch
from torch.nn import Module, Embedding, Linear, CrossEntropyLoss, Softmax
import math

class Net2(Module):
    def __init__(self,
                 num_input_classes=256,
                 embedding_dim=8,
                 context_length=64,
                 num_hidden=256,
                 num_output_classes=256):
        super().__init__()
        self.num_input_classes = num_input_classes # number of classes
        self.embedding_dim = embedding_dim # dimension of token embedding
        self.context_length = context_length # context window length
        self.num_hidden = num_hidden # number of hidden neurons
        self.num_output_classes = num_output_classes
        self.embedding = Embedding(self.num_input_classes, self.embedding_dim)
        self.encoder1 = Linear(self.embedding_dim*self.context_length, self.num_hidden)
        self.encoder2 = Linear(self.embedding_dim*self.context_length, self.num_hidden)
        self.decoder1 = Linear(self.num_hidden, self.num_output_classes)
        self.decoder2 = Linear(self.num_hidden, self.num_output_classes)
        self.criterion = CrossEntropyLoss(reduction='none')
        self.softmax = Softmax(dim=-1)

    def name(self):
        return f"net0({self.num_input_classes},{self.embedding_dim},{self.context_length},{self.num_hidden},{self.num_output_classes})"

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
        p = torch.randn(x.shape[:-1]+(self.num_hidden,),device=self.encoder1.weight.device)
        x = p*self.encoder1(x) + (1.0-p)*self.encoder2(x) # x.shape == [..., self.num_hidden]
        x = torch.sigmoid(x) # x.shape == [..., self.num_hidden]
        p = torch.randn(x.shape[:-1]+(self.num_output_classes,),device=self.encoder1.weight.device)
        x = p*self.decoder1(x) + (1.0-p)*self.decoder2(x) # x.shape == [..., self.num_output_classes]
        loss = self.criterion(x,y)/math.log(2)
        return loss

    def probs(self, X):
        L = self.context_length
        K = self.embedding_dim
        x = self.embedding(X[:,-self.context_length:]) # x.shape == [B, L, K]
        x = x.view(-1,self.context_length*self.embedding_dim) # x.shape == [..., self.context_length*self.embedding_dim]
        p = torch.randn(x.shape[:-1]+(self.num_hidden,),device=self.encoder1.weight.device)
        x = p*self.encoder1(x) + (1.0-p)*self.encoder2(x) # x.shape == [..., self.num_hidden]
        x = torch.sigmoid(x) # x.shape == [..., self.num_hidden]
        p = torch.randn(x.shape[:-1]+(self.num_output_classes,),device=self.encoder1.weight.device)
        x = p*self.decoder1(x) + (1.0-p)*self.decoder2(x) # x.shape == [..., self.num_output_classes]
        P = self.softmax(x) # P.shape == [..., self.num_output_classes]
        return P
