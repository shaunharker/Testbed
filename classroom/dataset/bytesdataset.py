import os
from pathlib import Path
from random import randrange
import numpy as np
import torch


class BytesDataset:
    def __init__(self,
                 path=None,
                 mode="random",
                 stride=1):
        if path is None:
            path = f"/home/{os.environ.get('USER')}/data/gutenberg.utf8"
        self.path = path
        self.mode = mode
        self.stride = stride
        self.n_bytes = Path(self.path).stat().st_size
        self.data = np.memmap(path, mode='r')
        self.pos = 0

    def batch(self, batch_size, example_length, pos=None):
        if pos is not None:
            self.pos = pos
        if self.mode == "jitter":
            if self.pos + 1024*batch_size > self.n_bytes:
                self.pos = self.pos % (1024*batch_size)
            result = torch.tensor(self.data[self.pos:self.pos+1024*batch_size].reshape(batch_size, 1024)[:, :example_length], dtype=torch.long, device='cuda')
            self.pos += 1024*batch_size + randrange(example_length)
            return result
        elif self.mode == "sequential":
            if self.pos + batch_size*self.stride > self.n_bytes:
                self.pos = self.pos % (batch_size*self.stride)
            example = lambda n: self.data[n:n+example_length]
            examples = [example(self.pos + idx*self.stride) for n in range(batch_size)]
            result = np.stack(examples)
            self.pos += batch_size*self.stride
            return torch.tensor(result, dtype=torch.long, device='cuda')
        elif self.mode == "random":
            example = lambda n: self.data[n:n+example_length]
            rand_pos = lambda: randrange(self.n_bytes-example_length)
            examples = [example(rand_pos()) for _ in range(batch_size)]
            result = np.stack(examples)
            return torch.tensor(result, dtype=torch.long, device='cuda')

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['data']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.data = np.memmap(self.path, mode='r')
