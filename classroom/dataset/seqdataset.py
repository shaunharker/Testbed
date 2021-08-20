import os
from random import randrange
import numpy as np
import threading
from pathlib import Path
import torch
import types
import time
from multiprocessing import Process, Queue, Event

class SeqDataset:
    def __init__(self,
                 path=None):
        if path is None:
            path = f"/home/{os.environ.get('USER')}/data/gutenberg.utf8"
        self.path = path
        self.n_bytes = Path(path).stat().st_size

    def __len__(self):
        return self.n_bytes // 1024

    def __getitem__(self, idx):
        return np.fromfile(path, dtype=np.uint8, count=1024, offset=1024*idx)
