import os
from pathlib import Path
from random import randrange
import numpy as np
import torch
import os
from .utf8 import utf8decode, utf8encode

user = os.environ["USER"]


class GutenbergBytesDataset:
    def __init__(self, path=None, device='cuda'):
        if path is None:
            path = f"/home/{user}/data/gutenberg.utf8"
        self.path = path
        self.device = device
        self.decode = utf8decode
        self.encode = utf8encode
        self._load()

    def batch(self, batch_size, example_length, offset=None):
        get_example = lambda: (lambda offset: self.data[offset:offset+example_length])(randrange(self.n_bytes-example_length))
        es = [get_example() for _ in range(batch_size)]
        return torch.tensor(
            np.stack(es).reshape(batch_size, example_length),
            dtype=torch.long,
            device=self.device)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['data']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load()

    def _load(self):
        self.n_bytes = Path(self.path).stat().st_size
        self.data = np.memmap(self.path, dtype=np.uint8, mode='r', offset=0)
