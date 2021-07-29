import os
from random import randrange
import numpy as np
import threading

class ByteDataset:
    def __init__(self,
                 path=None):
        if path is None:
            path = f'/home/{os.environ.get('USERNAME')}/data/gutenberg.1024.utf8'
        self.path = path
        self.n_bytes = Path(path).stat().st_size
        self._refresh()
        self.data = self.page
        self.thread = threading.Thread(target=self._refresh)
        self.thread.start()

    def batch(self, batch_size, example_length):
        assert example_length <= 512
        examples_left = self.data.shape[0]
        if examples_left < 2**15:
            self.thread.join()
            self.data = np.concatenate((self.data, self.page))
            self.thread = threading.Thread(target=self._refresh)
            self.thread.start()
        get_example = lambda idx: (lambda a, b, c: a[b:b+c])(
            self.data[idx], randrange(1024-example_length), example_length)
        result = np.stack(get_example(idx) for idx in range(batch_size))
        self.data = self.data[batch_size:,:]
        return result

    def _refresh(self):
        self.page = np.fromfile(self.path, dtype=np.ubyte, count=2**25, sep='',
                    offset= 1024*(randrange(self.n_bytes - 2**25)//1024)).reshape(-1,1024)
