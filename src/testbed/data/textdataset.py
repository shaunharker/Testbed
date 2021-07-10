import torch
import math
from random import randrange, randint
from ..util import default_device, decode_broken_utf8
from pathlib import Path
import numpy as np

class TextDataset:
    def __init__(self,
                 filename='/home/sharker/data/corpus.utf8.txt',
                 example_length=64):
        self.set_data_path(filename)
        self.set_example_length(example_length)
        self.count = 0
        self.cache_size = 65536
        self._cache_data()

    def set_data_path(self, filename):
        self.filename = filename
        self.file_length = Path(self.filename).stat().st_size
        self.data = None

    def set_example_length(self, example_length):
        self.example_length = example_length

    def __getitem__(self, idx):
        self.count += 1
        if self.count == self.cache_size:
            if self.cache_size < 2**30:
                self.cache_size = 2**30
                print(f"Increased TextDataset cache_size to {self.cache_size}.")
            self.count = 0
            self._cache_data()
        offset = idx
        while offset%self.run_length > self.run_length - self.example_length:
            offset = randrange(len(self))
        return self.data[offset:offset+self.example_length]

    def __len__(self):
        return self.data_length - self.example_length

    def random_text_snippet(self):
        idx = randrange(len(self))
        return self.inspect(idx)

    def inspect(self, idx):
        return decode_broken_utf8(self[idx])

    def _cache_data(self):
        self.run_length = 65536//self.example_length * self.example_length
        self.num_runs = self.cache_size//self.run_length
        offsets = [randrange(self.file_length-self.run_length) for _ in range(self.num_runs)]
        self.data = (
            torch.as_tensor(
                np.concatenate(
                    [np.fromfile(
                        self.filename,
                        dtype=np.ubyte,
                        count=self.run_length,
                        sep='',
                        offset=offset)
                    for offset in offsets])))
        self.data_length=len(self.data)
        print(f"Loaded {self.data_length} bytes of training data.")
