import torch
import math
from random import randrange, randint
from ..util import default_device, decode_broken_utf8
from pathlib import Path
import numpy as np

class TextDataset:
    def __init__(self,
                 filename='/home/sharker/data/corpus.utf8.txt',
                 N=64):
        self.set_data_path(filename)
        self.set_example_length(N)

    def cache_data(self):
        self.data = torch.as_tensor(np.fromfile(self.filename, dtype=np.ubyte))

    def set_data_path(self, filename):
        self.filename = filename
        self.data_len = Path(self.filename).stat().st_size
        self.data = None

    def set_example_length(self, N):
        self.N = N

    def __getitem__(self, idx):
        if self.data is None:
            offset = idx # randrange(self.data_len-self.N)
            return torch.as_tensor(np.fromfile(self.filename, dtype=np.ubyte,
                count=self.N, sep='', offset=offset))
        else:
            offset = idx # randrange(self.data_len-self.N)
            return self.data[offset:offset+self.N]

    def __len__(self):
        return self.data_len - self.N

    def random_text_snippet(self):
        idx = randrange(len(self))
        return self.inspect(idx)

    def inspect(self, idx):
        return decode_broken_utf8(self[idx])
