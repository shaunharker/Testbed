import os
from pathlib import Path
from random import randrange
import numpy as np
import os
import torch
import stockfish as Stockfish
import chess

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


def game_targets(game):
    def look(board):
        piece_encoding = {'.': 0, 'K': 1, 'Q': 2, 'N': 3, 'B': 4, 'R': 5, 'P': 6,
                          'k': 7, 'q': 8, 'n': 9, 'b': 10, 'r': 11, 'p': 12}
        fen = (board.fen().split()[0].replace("/", '').replace("8", "44")
            .replace("7", "43").replace("6", "33").replace("5", "32")
            .replace("4", "22").replace("3", "21").replace("2", "11")
            .replace("1", "."))
        return torch.tensor([piece_encoding[c] for c in fen], dtype=torch.long,  device="cuda")

    def hotloop(move, legal):
        n = len(move) + 1
        action_target = torch.zeros([n, 256], dtype=torch.long, device="cuda")
        for d in range(len(move)):
            c = move[d]
            for m in legal:
                if len(m) > d:
                    action_target[d, ord(m[d])] = 1
            legal = [m for m in legal if len(m) > d and m[d] == c]
        return action_target

    moves = game.split()
    N = len(game.strip()) + 1 # add one for newline preamble
    vision_target = torch.zeros([N,64], dtype=torch.long, device="cuda")
    action_target = torch.zeros([N,256], dtype=torch.long, device="cuda")
    board = chess.Board()
    idx = 0
    legal = [board.san(m) for m in board.legal_moves]
    for move in moves:
        n = len(move) + 1
        vision_target[idx:idx+n,:] = look(board).reshape([1,-1])
        action_target[idx:idx+n,:] = hotloop(move, legal)
        board.push_san(move)
        legal = [board.san(m) for m in board.legal_moves]
        if len(legal) > 0:
            action_target[idx+n-1, ord(" ")] = 1
        else:
            action_target[idx+n-1, ord("\n")] = 1
        idx += n
    torchify = lambda x: torch.tensor(np.frombuffer(bytes(x, encoding='utf8'),
        dtype=np.uint8), dtype=torch.long, device="cuda")
    seq_input = torchify("\n" + game.strip())
    seq_target = torchify(game.strip() + ("\n" if len(legal)==0 else " "))
    return seq_input, seq_target, vision_target, action_target


def training_batch():
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
        game = (game + " " + move).strip()
        if board.can_claim_draw():
            break
        ply += 1
        if ply > 512:
            break
    # if not a checkmate, remove tail after last capture or pawn advance
    revmoves = list(reversed(game.split()))
    keep = False
    moves = []
    for move in revmoves:
        if "x" in move:
            keep = True
        if not any(move.startswith(c) for c in ["N", "Q", "K", "B", "R"]):
            keep = True
        if keep:
            moves.append(move)
    moves = list(reversed(moves))
    moves = moves[:512]
    game = ' '.join(reversed(moves))
    return game_targets(game)
