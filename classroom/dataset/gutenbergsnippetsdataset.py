import os
from pathlib import Path
from random import randrange
import numpy as np
import torch
import os
from .utf8 import utf8decode, utf8encode

user = os.environ["USER"]


class GutenbergSnippetsDataset:
    def __init__(self, path=None, line=None, device='cuda'):
        if path is None:
            if line is None:
                line = 1024
            path = f"/home/{user}/data/gutenberg.{line}.utf8"
        self.path = path
        self.line = line
        self.device = device
        self.decode = utf8decode
        self.encode = utf8encode
        self._load()

    def batch(self, batch_size, example_length, offset = None):
        L = self.line or 2*example_length
        sz = batch_size*L
        offset = offset if offset is not None else randrange(self.n_bytes)
        batch_offset = offset%(self.n_bytes//sz)
        example_offset = (offset//(self.n_bytes//sz))%(L-example_length)
        return torch.tensor(self.data[sz*batch_offset:sz*batch_offset+sz].reshape(batch_size, L)[:,example_offset:example_offset+example_length], dtype=torch.long, device=self.device)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['data']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load()

    def _load(self):
        self.n_bytes = Path(self.path).stat().st_size
        self.data = np.memmap(self.path, mode='r')
