import torch
import numpy as np
import chess

bytes_to_tensor = lambda x: torch.tensor(np.frombuffer(
    bytes(x, encoding='utf8'), dtype=np.uint8),
    dtype=torch.long, device="cuda")

def targets(game):
    def look(board):
        piece_encoding = {'.': 0, 'K': 1, 'Q': 2, 'N': 3, 'B': 4, 'R': 5,
            'P': 6, 'k': 7, 'q': 8, 'n': 9, 'b': 10, 'r': 11, 'p': 12}
        fen = (board.fen().split()[0].replace("/", '').replace("8", "44")
            .replace("7", "43").replace("6", "33").replace("5", "32")
            .replace("4", "22").replace("3", "21").replace("2", "11")
            .replace("1", "."))
        return torch.tensor([piece_encoding[c] for c in fen],
            dtype=torch.long,  device="cuda")

    def chunk(move, legal):
        n = len(move) + 1
        action_target_chunk = torch.zeros([n, 256], dtype=torch.long,
            device="cuda")
        for d in range(len(move)):
            c = move[d]
            for m in legal:
                if len(m) > d:
                    action_target_chunk[d, ord(m[d])] = 1
            legal = [m for m in legal if len(m) > d and m[d] == c]
        return action_target_chunk

    moves = game.split()
    N = len(game.strip()) + 1 # add one for newline preamble
    visual_target = torch.zeros([N,64], dtype=torch.long, device="cuda")
    action_target = torch.zeros([N,256], dtype=torch.long, device="cuda")
    board = chess.Board()
    idx = 0
    legal = [board.san(m) for m in board.legal_moves]
    for move in moves:
        n = len(move) + 1
        visual_target[idx:idx+n,:] = look(board).reshape([1,-1])
        action_target[idx:idx+n,:] = chunk(move, legal)
        board.push_san(move)
        legal = [board.san(m) for m in board.legal_moves]
        if len(legal) > 0:
            action_target[idx+n-1, ord(" ")] = 1
        else:
            action_target[idx+n-1, ord("\n")] = 1
        idx += n

    seq_input = bytes_to_tensor("\n" + game.strip())
    seq_target = bytes_to_tensor(game.strip() +
        ("\n" if len(legal)==0 else " "))
    return seq_input, seq_target, visual_target, action_target
