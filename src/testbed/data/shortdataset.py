import torch
import math
import time
from random import randrange, randint
from ..util import default_device, decode_broken_utf8
from pathlib import Path
import numpy as np
import threading


class ShortDataset:
    def __init__(self,
                 filename='/home/sharker/data/gpt2tokenized.npy',
                 example_length=64,
                 dataline_size=2**20,
                 max_gpu_mem=2**30):
        self.set_data_path(filename)
        self.set_example_length(example_length)
        self.dataline_size = dataline_size
        self.max_gpu_mem = max_gpu_mem
        self.count = 0
        self.loader = threading.Thread(target=self._loading_daemon, daemon=True)
        self.lock = threading.Lock()
        self._cache_lines(1)
        self.loader.start()

    def set_data_path(self, filename):
        self.filename = filename
        self.file_length = Path(self.filename).stat().st_size
        self.data = None

    def set_example_length(self, example_length):
        self.example_length = example_length

    def __getitem__(self, idx):
        self.count += 1
        offset = randrange(len(self))
        while (offset + self.example_length) % self.dataline_size < self.example_length:
            offset = randrange(len(self))
        self.lock.acquire()
        item = self.data[offset:offset+self.example_length]
        self.lock.release()
        return item

    def __len__(self):
        return len(self.data) - self.example_length

    def _data_line(self):
        linebytes = 2*self.dataline_size
        offset = linebytes * randrange((self.file_length - linebytes)//linebytes) + 128 # +128 to skip numpy save header
        try:
            result = np.fromfile(
                self.filename,
                dtype=np.uint16,
                count=self.dataline_size,
                offset=offset)
        except:
            self.filename = "/home/ubuntu/data/gpt2tokenized.npy"
            result = np.fromfile(
                self.filename,
                dtype=np.uint16,
                count=self.dataline_size,
                offset=offset)
        return result

    def _cache_lines(self, num_lines=256):
        newdata = (
            torch.as_tensor(
                np.concatenate(
                    [self._data_line().astype(np.int32) for _ in range(num_lines)])).to('cuda',non_blocking=True))
        self.lock.acquire()
        self.data = newdata
        self.num_lines = num_lines
        self.lock.release()
        print(f"Loaded {len(self.data)*2} bytes of training data.")

    def _refresh_line(self):
        line_no = randrange(self.num_lines)
        offset = self.dataline_size*line_no
        line = torch.as_tensor(self._data_line().astype(np.int32)).to('cuda',non_blocking=True)
        self.lock.acquire()
        self.data[offset:offset+self.dataline_size] = line
        self.lock.release()

    def _loading_daemon(self):
        count = self.count
        i=1
        max_i = math.floor(math.log(self.max_gpu_mem/(2*self.dataline_size))/math.log(2)) - 1
        while True:
            time.sleep(.1)
            if i <= max_i and self.count > len(self) // 2**10:
                self._cache_lines(num_lines=2**i)
                i = i + 1
            if self.count > count:
                self._refresh_line()
                count = self.count
