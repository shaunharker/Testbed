import torch
import math
import time
from random import randrange, randint
from ..util import default_device, decode_broken_utf8
from pathlib import Path
import numpy as np
import threading

def default_utf8_path():
    return f'/home/{os.environ.get('USERNAME')}/data/corpus.utf8.txt'

def filesize_in_bytes(filename):
    return Path(filename).stat().st_size

def random_utf8_sequence(n_bytes=128, filename='/home/sharker/data/corpus.utf8.txt'):
    N = filesize_in_bytes(filename)
    offset = randrange(N - n_bytes)
    result = np.fromfile(
        filename,
        dtype=np.ubyte,
        count=n_bytes,
        sep='',
        offset=offset)
    return result

def load_corpus_bytes(filename='/home/sharker/data/corpus.utf8.txt'):
    if filename is None:
        filename=f'/home/{os.environ.get('USERNAME')}/data/corpus.utf8.txt',
    with open(filename, 'rb') as infile:
        data = infile.read()
    return data

class ByteDataset:
    def __init__(self,
                 filename=None,
                 example_length=64,
                 max_cache=2**30,
                 dataline_size=2**20):
        if filename is None:
            filename = default_utf8_path()
        self.kwargs = {
            "filename": filename,
            "example_length": example_length,
            "max_cache": max_cache,
            "dataline_size": dataline_size
        }
        self.set_data_path(filename)
        self.set_example_length(example_length)
        self.shuffle = shuffle
        self.dataline_size = dataline_size
        self.max_cache = max_cache
        self.count = 0
        self.loader = threading.Thread(target=self._loading_daemon, daemon=True)
        self.lock = threading.Lock()
        self._cache_lines(1)
        self.loader.start()

    def set_data_path(self, filename):
        self.filename = filename
        self.file_length = filesize_in_bytes(filename)
        self.data = None

    def set_example_length(self, example_length):
        self.example_length = example_length

    def __getitem__(self, idx):
        self.count += 1
        if self.shuffle:
            offset = randrange(len(self))
            while (offset + self.example_length) % self.dataline_size < self.example_length:
                offset = randrange(len(self))
        else:
            offset = self.count
        self.lock.acquire()
        item = self.data[offset:offset+self.example_length]
        self.lock.release()
        return item

    def __len__(self):
        return len(self.data) - self.example_length

    def random_text_snippet(self):
        idx = randrange(len(self))
        return self.inspect(idx)

    def inspect(self, idx):
        return decode_broken_utf8(self[idx])

    def _data_line(self, line_no=None):
        if line_no is None:
            offset = self.dataline_size * randrange((self.file_length-self.dataline_size)//self.dataline_size)
        else:
            offset = (line_no * self.dataline_size) % len(self)
        try:
            result = np.fromfile(
                self.filename,
                dtype=np.ubyte,
                count=self.dataline_size,
                sep='',
                offset=offset)
        except:
            self.filename = "/home/ubuntu/data/corpus.utf8.txt"
            result = np.fromfile(
                self.filename,
                dtype=np.ubyte,
                count=self.dataline_size,
                sep='',
                offset=offset)
        return result

    def _cache_lines(self, num_lines=2**10):
        newdata = (
            torch.as_tensor(
                np.concatenate(
                    [self._data_line() for _ in range(num_lines)])).to('cuda',non_blocking=True))
        self.lock.acquire()
        self.data = newdata
        self.num_lines = num_lines
        self.lock.release()
        print(f"Loaded {len(self.data)} bytes of training data.")

    def _refresh_line(self, line_no=None):
        if line_no is None:
            line_no = randrange(self.num_lines)
        offset = self.dataline_size*line_no
        line = torch.as_tensor(self._data_line()).to('cuda',non_blocking=True)
        self.lock.acquire()
        self.data[offset:offset+self.dataline_size] = line
        self.lock.release()

    def _loading_daemon(self):
        count = self.count
        i=1
        while True:
            time.sleep(.1)
            if i <= 10 and self.count > len(self) // 2**10:
                self._cache_lines(num_lines=2**i)
                i = i + 1
            if self.count > count:
                self._refresh_line()
                count = self.count
