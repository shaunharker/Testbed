import os
from pathlib import Path
from random import randrange
import numpy as np
import os
import torch
from stockfish import Stockfish
import chess
import types
from .targets import targets

def utf8encode(char_sequence):
    if type(char_sequence) == types.GeneratorType:
        def stream():
            for c in char_sequence:
                for b in bytes(c, encoding='utf8'):
                    yield b
        result = stream()
    else:
        result = bytes(char_sequence, encoding='utf8')
    return result

def utf8decode(byte_sequence):
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


class ChessDataset:
    def __init__(self, path=None, device='cuda'):
        if path is None:
            user = os.environ["USER"]
            path = f"/home/{user}/data/standard-chess.utf8"
        self.path = path
        self.device = device
        self.decode = utf8decode
        self.encode = utf8encode
        self._load()
        with open("/home/sharker/data/standard-chess.utf8") as infile:
            book = list(infile.readlines())
        self.book = sorted(book)
        self.in_book = lambda s: book[bisect.bisect_left(book, s)].startswith(s)

    def batch(self, batch_size, example_length):
        def adjust_offset(offset):
            """
            return next newline position after offset
            """
            return np.where(self.data[offset:offset+10000] == 10)[0][0] + offset
        def get_example():
            offset = self.n_bytes
            while offset + example_length >= self.n_bytes:
                offset = adjust_offset(randrange(self.n_bytes-example_length))
            return self.data[offset:offset+example_length]
        es = [get_example() for _ in range(batch_size)]
        return torch.tensor(
            np.stack(es).reshape(batch_size, example_length),
            dtype=torch.long,
            device=self.device)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['data']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load()

    def _load(self):
        self.n_bytes = Path(self.path).stat().st_size
        self.data = np.memmap(self.path, dtype=np.uint8, mode='r', offset=0)

    def bookgame(self, max_plies=800):
        game = self.book[randrange(len(self.book))]
        return ' '.join(game.split()[:max_plies])

    def stockfishgame(self, max_plies=800):
        engine = Stockfish()
        board = chess.Board()
        ply = 0
        game = ""
        while True:
            engine.set_fen_position(fen_position = board.fen())
            move = engine.get_best_move_time(time=1.0)
            if move is None:
                break
            move = board.san(board.parse_uci(move))
            board.push_san(move)
            game = game + " " + move if len(game) > 0 else move
            if board.can_claim_draw():
                break
            ply += 1
            if ply >= max_plies:
                break
        # if not a checkmate, remove tail after last capture or pawn advance
        # revmoves = list(reversed(game.split()))
        # keep = False
        # moves = []
        # for move in revmoves:
        #     if "x" in move:
        #         keep = True
        #     if not any(move.startswith(c) for c in ["N", "Q", "K", "B", "R"]):
        #         keep = True
        #     if keep:
        #         moves.append(move)
        # moves = list(reversed(moves))
        return game
