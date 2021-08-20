import os
from random import randrange
import numpy as np
import threading
from pathlib import Path
import torch
import types
import time
from multiprocessing import Process, Queue, Event

class UTF8Dataset:
    def __init__(self,
                 path=None):
        if path is None:
            path = f"/home/{os.environ.get('USER')}/data/gutenberg.1024.utf8"
        self.path = path
        self.n_bytes = Path(self.path).stat().st_size
        self.mem = np.memmap(path, mode='r')
        self.start = 0
        self.cache = torch.tensor([], dtype=torch.long, device='cuda')
        self.end = 0
        self.pos = self.n_bytes # causes it to randomize

    def batch(self, batch_size, example_length):
        shift_index = self.pos % (1024 - example_length + 1)
        batch_index = self.pos // (1024 - example_length + 1)
        if 1024*(batch_index+1)*batch_size > self.n_bytes:
            batch_index = randrange(self.n_bytes//(1024*batch_size))
            shift_index = randrange(1024 - example_length + 1)
            self.pos = shift_index + (1024 - example_length + 1)*batch_index
        # print(shift_index, batch_index)
        if self.start != 1024*batch_index*batch_size or self.end != 1024*(batch_index+1)*batch_size:
            self.start = 1024*batch_index*batch_size
            self.end = 1024*(batch_index+1)*batch_size
            # print(f"{self.n_bytes}, {self.pos}, {self.start}, {self.end}, {batch_index}, {batch_size}")
            self.cache = torch.tensor(self.mem[self.start:self.end], dtype=torch.long, device='cuda').view(batch_size, 1024)
        result = self.cache[:,shift_index:shift_index+example_length].contiguous()
        self.pos += 1
        return result
