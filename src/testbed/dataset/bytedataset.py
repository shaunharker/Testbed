import os
from random import randrange
import numpy as np
import threading
from pathlib import Path
import torch
import types

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
            the number of examples in a batch
            currently must be less than or equal to 512
        example_length: int
            the number of bytes in an example
    NOTE:
        batch_size and example_length are part of the public interface,
        and do not change unless update is called. The dataset is tuned
        to efficiently produce batches of size [batch_size, example_length]
        with examples distributed as approximately iid uniform draws with
        replacement. If the user requests a smaller size (in one or both
        dimensions) this can be done less efficiently (by the ratio of elements)
        but does not disturb the cache. If the shape is larger in either dimension
        the cache is flushed and refilled which is not performant, but probably
        acceptable for occasional use.
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
                 example_length=None):
        self.worker = None
        assert batch_size is not None, "batch_size: int  required"
        assert example_length is not None, "example_length: int required"
        assert example_length <= 512, "example_length <= 512 required"
        self.update(path=path, batch_size=batch_size, example_length=example_length)

    def update(self, path=None, batch_size=None, example_length=None):
        if self.worker is not None:
            self.worker.join()
        if path is None:
            path = f"/home/{os.environ.get('USERNAME')}/data/gutenberg.1024.utf8"
        self.path = path
        if batch_size is not None:
            self.batch_size = batch_size
        if example_length is not None:
            self.example_length = example_length
        self.n_bytes = Path(path).stat().st_size
        self.cache = []
        self.cache_shape = (self.batch_size, self.example_length)
        self.worker = None
        self.cache_lock = threading.Lock()
        self.cache_available = threading.Event()
        self.cache_invalid = threading.Event()
        self.data = None

    def batch(self, batch_size=None, example_length=None):
        """
        public method batch(self, batch_size=None, example_length=None)
        ---------------------------------------------------------------
        DESC: This method returns a batch of batch_size examples
              of length example_length randomly drawn from the dataset
              Formally, approximate iid uniform sampling with replacement.
        ARGS:
            batch_size: int
                the number of examples in a batch
            example_length: int
                the number of bytes in an example
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
        self._renew_worker_thread(max(batch_size, self.batch_size),
                                  max(example_length, self.example_length))

        # Pull the result batch off the cache when it is available.
        self.cache_available.wait()
        with self.cache_lock:
            sample = self.cache.pop()[:batch_size,:example_length]
            if len(self.cache) == 0:
                self.cache_available.clear()

        # Order the worker to make the default kind of outputs
        self._renew_worker_thread(self.batch_size, self.example_length)
        return sample

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
        """
        private method _worker(self, batch_size, example_length)
        --------------------------------------------------------
        DESC:
            This is the code executed by worker threads. It eventually
            exits on its own unless it can't keep up with demand and
            never fills the cache up to the desired_cache_size.
        ARGS:
            batch_size: int
                the number of examples in a batch
            example_length: int
                the number of bytes in an example
        RETURNS:
            None
        """
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
        """
        private method ByteDataset._batch(self, batch_size, example_length)
        -------------------------------------------------------------------
        DESC: This is the code that randomly cuts examples
              from the shuffled stream of examples coming from
              the file and stacks them into tensors.
        ARGS:
            batch_size: int
                the number of examples in a batch
            example_length: int
                the number of bytes in an example
        RETURNS:
            result:
                TYPE: torch.Tensor
                SHAPE: [batch_size, example_length]
                DTYPE: torch.uint8
                DEVICE: 'cuda'
        """
        get_example = lambda idx: (lambda a, b, c: a[b:b+c])(
            self.data[idx], randrange(1024-example_length+1), example_length)
        result = np.stack([get_example(idx) for idx in range(batch_size)])
        self.data = self.data[batch_size:,:]
        return torch.tensor(result, dtype=torch.uint8, device='cuda')

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
        def is_header_byte(b):
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
                    yield bytes(word).decode('utf8')
        if type(byte_sequence) == types.GeneratorType:
            return stream()
        else:
            return ''.join(list(stream()))
