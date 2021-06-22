import torch
from random import randrange
from . import default_device

class TextDataset:
    def __init__(self,
                 filename='minicorpus.txt',
                 N=64,
                 B=64,
                 shuffle=True,
                 batch_first=True,
                 device=None):
        if device is None:
            device = default_device()
        self.device = device
        self.N = N
        self.B = B
        self.batch_first = batch_first
        with open('minicorpus.txt', 'r') as infile:
            self.text = infile.read()
        try:
            self.tokens = torch.load('minicorpus.pt').to(device)
        except:
            self.tokens = torch.tensor(list(bytes(self.text, 'utf-8'))).byte()
            torch.save(self.tokens, 'minicorpus.pt')
        D = len(self.tokens) // (N*B)
        self.D = D
        self.perm = list(range(self.D))
        self.ready = False

    def set_batch_size(self, B):
        self.B = B
        N = self.N
        D = len(self.tokens) // (N*B)
        self.D = D
        device = self.device
        if self.batch_first:
            self.batches = self.tokens[:D*B*N].view(B,D,N).transpose(0,1).contiguous().to(device)
        else:
            self.batches = self.tokens[:D*B*N].view(B,D,N).transpose(0,1).transpose(1,2).contiguous().to(device)

    def __getitem__(self, idx):
        if not self.ready:
            self.set_batch_size(self.B)
            self.ready = True
        idx = self.perm[idx]
        return self.batches[idx].long()

    def __len__(self):
        return self.D

    def set_permutation(self, perm):
        self.perm = perm

    def random_text_snippet(self, N):
        idx = randrange(len(self.text) - N)
        return self.text[idx:idx+N]

    def inspect(self, idx):
        batch = self[idx].tolist()
        return [decode_broken_utf8(example) for example in batch]
