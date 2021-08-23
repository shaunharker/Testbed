import os
from pathlib import Path
from random import randrange
import numpy as np
import torch


class BytesDataset:
    def __init__(self, path=None):
        self.path = path
        self._load()

    def batch(self, offset, batch_size, example_length):
        sz = batch_size*example_length
        idx = offset%(self.n_bytes-sz)
        return torch.tensor(self.data[idx:idx+sz], dtype=torch.long, device='cuda').view(batch_size, example_length)

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
