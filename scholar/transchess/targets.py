import torch
import numpy as np
from subprocess import Popen, PIPE
import json
from typing import List

def analyze(games: List[str]):
    proc = Popen(["/home/sharker/github/scholar/scholar/transchess/engine", "full"],
        stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)
    games = games.strip() + " ..."
    out, err = proc.communicate(input=games)
    return [json.loads(x) for x in out.splitlines()]

bytes_to_tensor = lambda x: torch.tensor(np.frombuffer(
    bytes(x, encoding='utf8'), dtype=np.uint8),
    dtype=torch.long, device="cuda")


class Chessboard:
    def __init__(self, game=""):
        self.data = analyze(game)[0]
        self.moves = game.split()

    def fen(self, ply=-1):
        return self.data["fen"][ply]

    def look(self, ply=-1):
        piece_encoding = {
            '.': 0, 'K': 1, 'Q': 2, 'N': 3,
            'B': 4, 'R': 5, 'P': 6, 'k': 7,
            'q': 8, 'n': 9, 'b': 10, 'r': 11, 'p': 12}
        dot = (self.fen(ply).split()[0]
            .replace("/", '').replace("8", "44")
            .replace("7", "43").replace("6", "33")
            .replace("5", "32").replace("4", "22")
            .replace("3", "21").replace("2", "11")
            .replace("1", "."))
        return torch.tensor([piece_encoding[c] for c in dot],
            dtype=torch.long,  device="cuda")

    def legal(self, ply=-1):
        return self.data["legal"][ply]

    def chunk(self, ply=-1):
        move = self.data["game"][ply]
        legal = self.data["legal"][ply]

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
        return action_target_chunk

def targets(game):
    moves = game.split()
    N = len(game.strip()) + 2
    seq_input = bytes_to_tensor("\n" + game.strip() + " ")
    seq_target = bytes_to_tensor(game.strip() + " ")
    visual_target = torch.zeros([N,64], dtype=torch.long, device="cuda")
    action_target = torch.zeros([N,256], dtype=torch.long, device="cuda")
    board = Chessboard(game.strip())
    idx = 0
    for ply, move in enumerate(moves):
        n = len(move) + 1
        visual_target[idx:idx+n,:] = board.look(ply).reshape([1,-1])
        action_target[idx:idx+n,:] = board.chunk(ply)
        idx += n
    visual_target[idx] = board.look(-1)
    for c in board.legal(-1):
        action_target[idx,ord(c[0])] = 1
    return seq_input, seq_target, visual_target, action_target
