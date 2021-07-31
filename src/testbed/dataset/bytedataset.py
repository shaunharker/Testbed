import os
from random import randrange
import numpy as np
import threading
from pathlib import Path
import torch
import types
import time

class ByteDataset:
    """
    class ByteDataset
    -----------------
    ARGS:
        path:
            TYPE: str
            DESC: the location of the dataset text file
                  Currently defaults to gutenberg.1024.utf8,
                  which is 14GB of Gutenberg text reencoded in utf8,
                  chopped up into length 1024 pieces, and shuffled.
        batch_size: int
            the default number of examples in a batch
        example_length: int
            the default number of bytes in an example
            currently must be less than or equal to 1024
            note that for example_length near 1024 there are
            far fewer batches in the entire dataset due to the
            fact there is only one way an example can fit in a
            size 1024 window rather than, say, the 512 ways for
            an example of length 512.
    TODO:
        * Find a way to gracefully support multiple cache sizes
          and be able to adapt to any adversarial sequence of batch requests
          with arguments (batch_size, example_length) with product small enough.
        * Create code to create files like gutenberg.1024.utf8 on demand for performance
          requirements. (e.g. gutenberg.256.utf8, gutenberg.65536.utf8)
        * Generalize the code so it can use gutenberg.X.utf8 for X=2**n
    """
    def __init__(self,
                 path=None,
                 batch_size=None,
                 example_length=None,
                 shuffle_blocks=True):
        self.worker = None
        assert batch_size is not None, "batch_size: int  required"
        assert example_length is not None, "example_length: int required"
        assert example_length <= 512, "example_length <= 512 required"
        self.update(path=path, batch_size=batch_size, example_length=example_length, shuffle_blocks=shuffle_blocks)

    def update(self, path=None, batch_size=None, example_length=None, shuffle_blocks=True):
        if self.worker is not None:
            self.worker.join()
        if path is None:
            if shuffle_blocks==True:
                path = f"/home/{os.environ.get('USER')}/data/gutenberg.1024.utf8"
            else:
                path = f"/home/{os.environ.get('USER')}/data/gutenberg.utf8"
        self.path = path
        if batch_size is not None:
            self.batch_size = batch_size
        if example_length is not None:
            self.example_length = example_length
        self.shuffle_blocks = shuffle_blocks
        self.n_bytes = Path(path).stat().st_size
        self.cache = []
        self.cache_shape = (self.batch_size, self.example_length)
        self.worker = None
        self.block_idx = 0
        self.lock = threading.Lock()
        self.batch_available = threading.Event()
        self.terminate_worker = threading.Event()
        self.data = np.empty((0,1024),dtype=np.uint8)

    def batch(self, batch_size=None, example_length=None):
        """
        public method batch(self, batch_size=None, example_length=None)
        ---------------------------------------------------------------
        DESC: This method returns a batch of batch_size examples
              of length example_length randomly drawn from the dataset
              in a way that approximates iid uniform sampling with replacement.
        ARGS:
            batch_size: int
                the number of examples in the returned batch
            example_length: int
                the number of bytes in an example in the returned batch
        RETURNS:
            result:
                TYPE: torch.Tensor
                SHAPE: [batch_size, example_length]
                DTYPE: torch.uint8
                DEVICE: 'cuda'
        """
        # Default arguments
        if batch_size is None:
            batch_size = self.batch_size
        if example_length is None:
            example_length = self.example_length

        # Validate arguments
        assert example_length <= 512, "example_length <= 512 required"

        # Order the worker to make these kind of outputs
        self._renew_worker_thread(batch_size, example_length)

        # Pull the result batch off the cache when it is available.
        self.batch_available.wait()
        with self.lock:
            sample = self.cache.pop()
            if len(self.cache) == 0:
                self.batch_available.clear()

        # Order the worker to make the default kind of outputs
        self._renew_worker_thread(batch_size, example_length)
        return sample.long()

    def _renew_worker_thread(self, batch_size, example_length):
        """
        private method _renew_worker_thread
        -----------------------------------
        DESC:
            If batch_size or example_length are non-default,
            toss the cache.
            In all cases, make sure a worker thread is left
            running with the argument parameters upon exit.
            The worker's job is to fill the cache.
        ARGS:
            batch_size: int
                the number of examples in a batch
            example_length: int
                the number of bytes in an example
        RETURNS:
            None
        """
        if (batch_size, example_length) != self.cache_shape:
            self.terminate_worker.set()
            if self.worker is not None:
                self.worker.join()
            with self.lock:
                self.cache = []
                self.cache_shape = (batch_size, example_length)
                self.terminate_worker.clear()
                self.batch_available.clear()
        if self.worker is None or not self.worker.is_alive():
            self.worker = threading.Thread(
                target=self._worker,
                args=(batch_size, example_length,))
            self.worker.start()

    def _worker(self, batch_size, example_length):
        """
        private method _worker(self, batch_size, example_length)
        --------------------------------------------------------
        DESC:
            This is the code executed by worker threads. It
        ARGS:
            batch_size: int
                the number of examples in a batch
            example_length: int
                the number of bytes in an example
        RETURNS:
            None
        """
        page_size = 2**20
        max_ctx = 1024
        n_examples = page_size // max_ctx
        max_cache = max(2**25, batch_size * max_ctx)
        with self.lock:
            if len(self.cache) * max_ctx * batch_size > max_cache:
                return
        assert example_length <= 512
        def randomload():
            offset = max_ctx*randrange(self.n_bytes - page_size)//max_ctx
            return np.fromfile(self.path, dtype=np.uint8, count=page_size,
                sep='', offset=offset).reshape(n_examples, max_ctx)
        def sequentialload():
            self.block_idx = self.block_idx % (self.n_bytes//max_ctx - n_examples)
            offset = max_ctx*self.block_idx
            self.block_idx += 1
            return np.fromfile(self.path, dtype=np.uint8, count=page_size,
                sep='', offset=offset).reshape(n_examples, max_ctx)
        if self.shuffle_blocks:
            load = randomload
        else:
            load = sequentialload
        while self.data.shape[0] < batch_size:
            self.data = np.concatenate((self.data, load()))
        def get_example(idx):
            offset = randrange(max_ctx-example_length+1)
            return self.data[idx][offset:offset+example_length]
        np_batch = np.stack([get_example(idx) for idx in range(batch_size)])
        self.data = self.data[batch_size:,:]
        while True:
            try:
                torch_batch = torch.tensor(np_batch, dtype=torch.uint8, device='cuda')
                break
            except:
                time.sleep(1.0)
                torch.cuda.empty_cache()
        with self.lock:
            self.cache.append(torch_batch)
            self.batch_available.set()

    @staticmethod
    def encode(char_sequence):
        """
        static method ByteDataset.encode(char_sequence)
        -----------------------------------------------
        ARGS:
            char_sequence:
                type: str | generator[char]
                desc: the string we will return the utf8 encoding for
        RETURNS:
            if type(char_sequence) == generator[char]:
                result:
                    TYPE: uint8 generator
                    DESC: a uint8 generator which yields the bytes
                          resulting from encoding the generator
                          char_sequence
            elif type(char_sequence) == str:
                result:
                    TYPE: bytes
                    DESC: the bytes resulting from encoding the
                          string char_sequence
        """
        if type(char_sequence) == types.GeneratorType:
            def stream():
                for c in char_sequence:
                    for b in bytes(c, encoding='utf8'):
                        yield b
            result = stream()
        else:
            result = bytes(char_sequence, encoding='utf8')
        return result

    @staticmethod
    def decode(byte_sequence):
        """
        static method ByteDataset.decode(byte_sequence)
        -----------------------------------------------
        DESC:
            a fault-tolerant utf8 decoder
        ARGS:
            byte_sequence:
                type: uint8 iterable (e.g. bytes)
                desc: a bytes iterable which is to be decoded as utf8 where
                      invalid utf8 is handled by ignoring invalid bytes
        RETURNS:
            if type(byte_sequence) == types.GeneratorType
                result:
                    TYPE: str generator
                    DESC: a str generator which yields the characters
                          of the string resulting from decoding s
            else:
                result:
                    TYPE: str
                    DESC: the string resulting from decoding s
        """
        def is_valid_utf8_byte(b):
            return b&0b11111000 != 0b11111000
        def is_payload_utf8_byte(b):
            return b&0b11000000 == 0b10000000
        def is_header_utf8_byte(b):
            return is_valid_utf8_byte(b) and not is_payload_utf8_byte(b)
        def char_width(b):
            if b&0b10000000 == 0:
                return 1
            elif b&0b11100000 == 0b11000000:
                return 2
            elif b&0b11110000 == 0b11100000:
                return 3
            elif b&0b11111000 == 0b11110000:
                return 4
            return None
        def stream():
            (word, width) = ([], 0)
            for b in byte_sequence:
                if is_header_utf8_byte(b):
                    (word, width) = ([b], char_width(b))
                elif is_payload_utf8_byte(b):
                    word.append(b)
                if len(word) == width:
                    try:
                        yield bytes(word).decode('utf8')
                    except:
                        # There are still undecodables we catch here.
                        # e.g. bytes(map(lambda x: int(x,base=2),['0b11000000', '0b10000000'])).decode('utf8') raises UnicodeDecodeError
                        pass
        if type(byte_sequence) == types.GeneratorType:
            return stream()
        else:
            return ''.join(list(stream()))
