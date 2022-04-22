import os
from pathlib import Path
from random import randrange
import numpy as np
import torch
import os
from .utf8 import utf8bitsdecode, utf8bitsencode


class GutenbergBitsDataset:
    def __init__(self, path=None, device='cuda'):
        if path is None:
            user = os.environ["USER"]
            path = f"/home/{user}/data/gutenberg.utf8"
        self.path = path
        self.device = device
        self.decode = utf8bitsdecode
        self.encode = utf8bitsencode
        self._load()

    def batch(self, batch_size, example_length):
        n_example_bytes = example_length//8 + 1
        get_example = lambda: (lambda byte_offset, bit_offset: np.unpackbits(self.data[byte_offset:byte_offset+n_example_bytes],
            bitorder='little')[bit_offset:bit_offset+example_length])(randrange(self.n_bytes-n_example_bytes), randrange(8))
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
        self.data = np.memmap(self.path, mode='r')
