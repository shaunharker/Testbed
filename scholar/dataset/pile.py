import os
import random
import json
import torch

from .utf8 import utf8decode, utf8encode

class ShuffledDataStreamer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.index = []
        self.line_lengths = []

        offset = 0
        with open(filepath, 'r') as f:
            for line in f:
                self.index.append(offset)
                line_length = len(line)
                self.line_lengths.append(line_length)
                offset += line_length

    def stream(self, start_line=None):
        if start_line is None:
            start_line = random.randint(0, len(self.index) - 1)

        yield from self._stream_from_line(start_line)

    def _stream_from_line(self, start_line):
        with open(self.filepath, 'r') as f:
            for line_number in range(start_line, len(self.index)):
                f.seek(self.index[line_number])
                text = json.loads(f.readline())['text']
                yield text

            for line_number in range(0, start_line):
                f.seek(self.index[line_number])
                yield json.loads(f.readline())['text']

    def get_line_length(self, line_number):
        return self.line_lengths[line_number]

    def line_count(self):
        return len(self.index)
    

def accumulate_bytes_until(gen, byte_size):
    accumulated = bytes()
    for item in gen:
        item_bytes = item.encode('utf-8')
        accumulated += item_bytes
        while len(accumulated) >= byte_size:
            yield accumulated[:byte_size]
            accumulated = accumulated[byte_size:]


class OldPileBytesDataset:
    def __init__(self, path=None, device='cuda'):
        if path is None:
            user = os.environ["USER"]
            path = f"/data/thepile/00.jsonl"
        self.path = path
        self.device = device
        self.decode = utf8decode
        self.encode = utf8encode
        self.streamer = ShuffledDataStreamer("/data/thepile/00.jsonl")
        self.reader = self.streamer.stream()

    def batch(self, batch_size, example_length):
        return torch.stack([torch.tensor([b for b in snippet], dtype=torch.long, device=self.device)
            for snippet, _ in zip(accumulate_bytes_until(self.reader, example_length), range(batch_size))])

def accumulator(gen, byte_size):
    accumulated = bytes()
    for item in gen:
        item_bytes = item.encode('utf-8')
        accumulated += item_bytes
        if len(accumulated) >= byte_size:
            offset = random.randint(0, len(accumulated) - byte_size)
            yield accumulated[offset:offset+byte_size]

class PileBytesDataset:
    def __init__(self, path=None, device='cuda'):
        if path is None:
            user = os.environ["USER"]
            path = f"/data/thepile/01.jsonl"
        self.path = path
        self.device = device
        self.decode = utf8decode
        self.encode = utf8encode
        self.streamer = ShuffledDataStreamer("/data/thepile/01.jsonl")
        self.reader = self.streamer.stream()

    def batch(self, batch_size, example_length):
        while True:
            try:
                result = torch.stack([torch.tensor([b for b in snippet], dtype=torch.long, device=self.device)
                    for snippet, _ in zip(accumulator(self.reader, example_length), range(batch_size))])
                return result
            except:
                self.reader = self.streamer.stream()
