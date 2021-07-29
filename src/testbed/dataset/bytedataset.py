import os
from random import randrange
import numpy as np
import threading
from pathlib import Path
import torch

class ByteDataset:
    def __init__(self,
                 path=None):
        if path is None:
            path = f"/home/{os.environ.get('USERNAME')}/data/gutenberg.1024.utf8"
        self.worker = None
        self.update(path=path)

    def update(self, path):
        if type(path) == dict:
            if "path" in path:
                path = path["path"]
            else:
                return {"path": self.path}
        if self.worker is not None:
            self.worker.join()
        self.path = path
        self.n_bytes = Path(path).stat().st_size
        self.cache = []
        self.cache_shape = None
        self.worker = None
        self.cache_lock = threading.Lock()
        self.cache_available = threading.Event()
        self.cache_invalid = threading.Event()
        self.data = None
        return {"path": self.path}

    def batch(self, batch_size, example_length):
        assert example_length <= 1024
        self.renew_worker_thread(batch_size, example_length)
        #print(self.cache_available, self.cache_available.wait(), len(self.cache))
        self.cache_available.wait()
        with self.cache_lock:
            #print(len(self.cache), self.cache_available)
            result = self.cache.pop()
            if len(self.cache) == 0:
                self.cache_available.clear()
        self.renew_worker_thread(batch_size, example_length)
        return result

    def renew_worker_thread(self, batch_size, example_length):
        if (batch_size, example_length) != self.cache_shape:
            self.cache_invalid.set()
            if self.worker is not None:
                self.worker.join()
            with self.cache_lock:
                self.cache = []
                self.cache_shape = (batch_size, example_length)
                self.cache_invalid.clear()
                self.cache_available.clear()
        if self.worker is None or not self.worker.is_alive():
            self.worker = threading.Thread(
                target=self._worker,
                args=(batch_size, example_length,))
            self.worker.start()

    def _worker(self, batch_size, example_length):
        page = lambda : np.fromfile(self.path, dtype=np.ubyte, count=2**25, sep='',
            offset= 1024*(randrange(self.n_bytes - 2**25)//1024)).reshape(-1,1024)
        if self.data is None:
            self.data = page()
        desired_cache_size = max(2**25//(batch_size * example_length), 1)
        while True:
            with self.cache_lock:
                if len(self.cache) > desired_cache_size:
                    break
            if self.cache_invalid.is_set():
                break
            examples_left = self.data.shape[0]
            if examples_left < batch_size:
                self.data = np.concatenate((self.data, page()))
            result = self._batch(batch_size, example_length)
            with self.cache_lock:
                self.cache.append(result)
                self.cache_available.set()

    def _batch(self, batch_size, example_length):
        get_example = lambda idx: (lambda a, b, c: a[b:b+c])(
            self.data[idx], randrange(1024-example_length+1), example_length)
        result = np.stack([get_example(idx) for idx in range(batch_size)])
        self.data = self.data[batch_size:,:]
        return torch.tensor(result,dtype=torch.long,device='cuda')
