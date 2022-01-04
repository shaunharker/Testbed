import os
from pathlib import Path
from random import randrange
import numpy as np
import torch
import os
from .gpt2 import gpt2decode, gpt2encode


class GutenbergGPT2Dataset:
    def __init__(self, path=None, device='cuda'):
        if path is None:
            user = os.environ["USER"]
            path = f"/home/{user}/data/gutenberg.gpt2.npy"
        self.path = path
        self.device = device
        self.decode = gpt2decode
        self.encode = gpt2encode
        self.offsets = []
        self._load()

    def push(self, offset):
        self.offsets.append(offset)

    def batch(self, batch_size, example_length):
        offset_fun = lambda: self.offsets.pop() if len(self.offsets) > 0 else randrange(self.n_tokens-example_length)
        get_example = lambda: (lambda offset: self.data[offset:offset+example_length])(offset_fun())
        es = [get_example() for _ in range(batch_size)]
        return torch.tensor(
            np.stack(es).reshape(batch_size, example_length).astype(np.int32),
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
        self.n_tokens = (Path(self.path).stat().st_size - 128)//2
        self.data = np.memmap(self.path, dtype=np.uint16, mode='r', offset=128)
