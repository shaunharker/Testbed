import os
from pathlib import Path
from random import randrange
import numpy as np
import os
from .utf8 import utf8decode, utf8encode


class ChessDataset:
    def __init__(self, path=None, device='cuda'):
        if path is None:
            user = os.environ["USER"]
            path = f"/home/{user}/data/standard-chess.utf8"
        self.path = path
        self.device = device
        self.decode = utf8decode
        self.encode = utf8encode
        self._load()

    def batch(self, batch_size, example_length):
        def adjust_offset(offset):
            """
            return next newline position after offset
            """
            return np.where(self.data[offset:offset+10000] == 10)[0][0] + offset
        def get_example():
            offset = self.n_bytes
            while offset + example_length >= self.n_bytes:
                offset = adjust_offset(randrange(self.n_bytes-example_length))
            return self.data[offset:offset+example_length]
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
