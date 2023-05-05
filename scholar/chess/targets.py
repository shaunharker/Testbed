import torch
import numpy as np
from subprocess import Popen, PIPE
import json
from typing import List
import time
import functools
from chessboard import Chessboard

bytes_to_tensor = lambda x: torch.tensor(np.frombuffer(
    bytes(x, encoding='utf8'), dtype=np.uint8),
    dtype=torch.long, device="cuda")


class TargetCalculator:
    def __init__(self, moves):
        board = Chessboard()
        self.moves = moves
        self.legal = [board.legal()]
        self.fen = [board.fen()]
        for move in moves:
            if board.move(move):
                self.legal.append(board.legal())
                self.fen.append(board.fen())
            else:
                raise ValueError(f"TargetCalculator: Invalid move {move}")

    def look(self, ply=-1):
        piece_encoding = {
            '.': 0, 'K': 1, 'Q': 2, 'N': 3,
            'B': 4, 'R': 5, 'P': 6, 'k': 7,
            'q': 8, 'n': 9, 'b': 10, 'r': 11, 'p': 12}
        dot = (self.fen[ply].split()[0]
            .replace("/", '').replace("8", "44")
            .replace("7", "43").replace("6", "33")
            .replace("5", "32").replace("4", "22")
            .replace("3", "21").replace("2", "11")
            .replace("1", "."))
        return torch.tensor([piece_encoding[c] for c in dot],
            dtype=torch.long,  device="cuda")


    def chunk(self, ply=-1):
        move = self.moves[ply]
        legal = self.legal[ply]

        n = len(move) + 1
        action_target_chunk = torch.zeros([n, 256], dtype=torch.long,
            device="cuda")
        for d in range(len(move)):
            c = move[d]
            for m in legal:
                if len(m) > d:
                    action_target_chunk[d, ord(m[d])] = 1
            legal = [m for m in legal if len(m) > d
                and m[d] == c]
        action_target_chunk[len(move), ord(' ')] = 1
        return action_target_chunk

def maketargets(game, seq_length):
    assert len(game.strip()) > seq_length
    moves = game.split()
    ply = 0
    sz = 1
    for move in moves:
        ply += 1
        sz += len(move) + 1
        if sz > seq_length:
            break
    moves = moves[:ply]
    game = ' '.join(moves)

    tc = TargetCalculator(moves)
    # print(seq_length, f"'{game}'")
    seq_input = bytes_to_tensor("\n" + game + " ")[:seq_length]
    seq_target = bytes_to_tensor(game + " ")[:seq_length]

    N = len(game) + 2
    visual_target = torch.zeros([N,64],
        dtype=torch.long, device="cuda")
    action_target = torch.zeros([N,256],
        dtype=torch.long, device="cuda")
    idx = 0
    for ply, move in enumerate(moves):
        n = len(move) + 1
        visual_target[idx:idx+n,:] = tc.look(ply).reshape([1,-1])
        action_target[idx:idx+n,:] = tc.chunk(ply)
        idx += n
    visual_target[idx] = tc.look(-1)
    for c in tc.legal[-1]:
        action_target[idx,ord(c[0])] = 1
    visual_target = visual_target[:seq_length]
    action_target = action_target[:seq_length]
    return (seq_input, seq_target,
        visual_target, action_target)
