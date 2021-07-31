import os
from random import randrange
import numpy as np
import threading
from pathlib import Path
import torch
import types
import time

class SeqByteDataset:
    """
    class SeqByteDataset
    --------------------
    ARGS:
        path:
            TYPE: str
            DESC: the location of the dataset text file
                  Currently defaults to gutenberg.utf8,
                  which is 14GB of Gutenberg text reencoded in utf8
        batch_size: int
        example_length: int
    """
    def __init__(self,
                 path=None,
                 batch_size=None,
                 example_length=None):
        assert batch_size is not None, "batch_size: int  required"
        assert example_length is not None, "example_length: int required"
        self.update(path=path, batch_size=batch_size, example_length=example_length)

    def update(self, path=None, batch_size=None, example_length=None):
        if path is None:
            path = f"/home/{os.environ.get('USERNAME')}/data/gutenberg.utf8"
        self.path = path
        if batch_size is not None:
            self.batch_size = batch_size
        if example_length is not None:
            self.example_length = example_length
        self.n_bytes = Path(path).stat().st_size
        self.offset = 0

    def batch(self, batch_size, example_length):
        """
        public method batch(self, batch_size, example_length)
        ---------------------------------------------------------------
        DESC: This method returns a batch of batch_size examples
              of length example_length each subsequent example
              intersecting by half their width (and thus every
              example being covered by its neighbor and every
              example_length substring being an example.)
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
        def examples():
            for _ in range(batch_size):
                if self.offset + example_length >= self.n_bytes: self.offset = 0
                yield torch.tensor(np.fromfile(self.path, dtype=np.uint8, count=example_length,
                    offset=self.offset).reshape(1,example_length), dtype=torch.long, device='cuda')
                self.offset += example_length // 2
        return torch.cat([x for x in examples()])


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
