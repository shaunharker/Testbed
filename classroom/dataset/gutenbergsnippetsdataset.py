import os
from pathlib import Path
from random import randrange
import numpy as np
import torch
import os

user = os.environ["USER"]


class GutenbergSnippetsDataset:
    def __init__(self, path=None, device='cuda'):
        self.path = path or f"/home/{user}/data/gutenberg.1024.utf8"
        self.device = device
        self._load()

    def batch(self, batch_size, example_length, offset = None):
        sz = batch_size*1024
        offset = offset or randrange(self.n_bytes)
        batch_offset = offset%(self.n_bytes//sz)
        example_offset = (offset//(self.n_bytes//sz))%(1024-example_length)
        return torch.tensor(self.data[1024*batch_offset:1024*batch_offset+sz].reshape(batch_size, 1024)[:,example_offset:example_offset+example_length], dtype=torch.long, device=self.device)

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
