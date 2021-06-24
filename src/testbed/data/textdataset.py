import torch
import math
from random import randrange, randint
from ..util import default_device, decode_broken_utf8

class TextDataset:
    def __init__(self,
                 filename='/home/sharker/data/minicorpus.txt',
                 N=64,
                 B=64,
                 shuffle=True,
                 device=None):
        self.filename = filename
        self.N = N
        self.B = B
        self.shuffle = shuffle
        if device is None:
            device = default_device()
        self.device = device
        with open(filename, 'r') as infile:
            self.text = infile.read()
        self.load_tokens()
        if shuffle:
            self.perform_shuffle(N)
        self.set_batch_size(B)

    def state_dict(self):
        return {"filename": self.filename,
                "N": self.N,
                "B": self.B,
                "shuffle": self.shuffle }

    def load_tokens(self):
        try:
            self.tokens = torch.load(self.filename + '.pt')
        except:
            self.tokens = torch.tensor(list(bytes(self.text, 'utf-8'))).byte()
            print("Saving dataset in .pt format")
            torch.save(self.tokens, self.filename + '.pt')

    def perform_shuffle(self, N, recompute_batches=True):
        """
        The goal is to mix up all the length N pieces without
        messing up the length N pieces themselves.
        It shrinks a bit, so we refresh to the original state once it
        gets to 70% of original size.

        Set recompute_batches to False in order to prevent set_batch_size
        from being called (e.g. you want to change the batch_size B anyway
        and will call it yourself)

        Warning: mildly clever
        """
        if len(self.tokens) < .7 * len(self.text):
            self.load_tokens()
        K = len(self.tokens) // N
        a = randint(int(math.sqrt(K)/2), int(2*math.sqrt(K)))
        b = K // a
        self.tokens = self.tokens[:a*b*N].view(a,b,N).transpose(0,1).contiguous().view(-1)[:a*b*N]
        if recompute_batches:
            self.set_batch_size(self.B)

    def set_example_length(self, N):
        self.N = N
        self.perform_shuffle(N)

    def set_batch_size(self, B):
        self.B = B
        N = self.N
        D = len(self.tokens) // (N*B)
        self.D = D
        device = self.device
        self.batches = self.tokens[:D*B*N].view(D,B,N).contiguous().to(device)
        self.perm = torch.randperm(D)

    def __getitem__(self, idx):
        return self.batches[self.perm[idx]].long()

    def __len__(self):
        return self.D

    def random_text_snippet(self, N):
        idx = randrange(len(self.text) - N)
        return self.text[idx:idx+N]

    def inspect(self, idx):
        batch = self[idx].tolist()
        return [decode_broken_utf8(example) for example in batch]
