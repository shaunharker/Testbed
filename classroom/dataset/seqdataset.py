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
        return np.fromfile(path, dtype=np.uint8, count=1024,
                        offset=1024*idx)

    @staticmethod
    def encode(char_sequence):
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
