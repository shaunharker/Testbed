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
        self.thread = None
        self.update(path=path)

    def update(self, path):
        if type(path) == dict:
            if "path" in path:
                path = path["path"]
            else:
                return {"path": self.path}
        if self.thread is not None:
            self.thread.join()
        self.path = path
        self.n_bytes = Path(path).stat().st_size
        self._refresh()
        self.data = self.page
        self.cache = []
        self.cache_thread = None
        self.cache_lock = threading.Lock()
        return {"path": self.path}

    def batch(self, batch_size, example_length):
        assert batch_size * example_length <= 2**24 # TODO: relax this requirement
        def ensure_cache_thread_running():
            if self.cache_thread is None or not self.cache_thread.is_alive():
                self.cache_thread = threading.Thread(
                    target=self._cache,
                    args=(batch_size, example_length,))
                self.cache_thread.start()
        def try_to_return_cache():
            self.cache_lock.acquire()
            result = self.cache.pop()
            self.cache_lock.release()
            if tuple(result.shape) != (batch_size, example_length):
                self.cache_thread.join()
                self.cache = []
                ensure_cache_thread_running()
                self.cache_thread.join()
                return self.cache.pop()
            else:
                if len(self.cache)*batch_size*example_length < 2**24:
                    ensure_cache_thread_running()
                return result
        if len(self.cache) > 0:
            return try_to_return_cache()
        elif len(self.cache) == 0:
            ensure_cache_thread_running()
            self.cache_thread.join()
            return try_to_return_cache()

    def _cache(self, batch_size, example_length):
        while len(self.cache) * batch_size * example_length < 2**25:
            examples_left = self.data.shape[0]
            if examples_left < 2**15:
                self._refresh()
                self.data = np.concatenate((self.data, self.page))
            result = self._batch(batch_size, example_length)
            self.cache_lock.acquire()
            self.cache.append(result)
            self.cache_lock.release()

    def _batch(self, batch_size, example_length):
        get_example = lambda idx: (lambda a, b, c: a[b:b+c])(
            self.data[idx], randrange(1024-example_length), example_length)
        result = np.stack(get_example(idx) for idx in range(batch_size))
        self.data = self.data[batch_size:,:]
        return torch.tensor(result,dtype=torch.long,device='cuda')

    def _refresh(self):
        self.page = np.fromfile(self.path, dtype=np.ubyte, count=2**25, sep='',
                    offset= 1024*(randrange(self.n_bytes - 2**25)//1024)).reshape(-1,1024)
