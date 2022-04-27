import random
import string
import chess
from stockfish import Stockfish
from IPython.display import display, Image, HTML
import asyncio

class ChessApp:
    def __init__(self, game=""):
        self.dispid = ''.join(
            random.choice(string.ascii_uppercase +
                          string.ascii_lowercase + string.digits) for _ in range(16))

        self.all_games = []
        self.game = game.strip()
        self.moves = [move.split('.')[1] if len(move.split('.')) > 1 else move
                 for move in game.split() if move[-1] != '.']
        self.board = chess.Board()
        for move in self.moves:
            self.board.push_san(move)
        self.board_display = display(self.board, display_id=f"board{self.dispid}")
        self.html = HTML()
        self.html_display = display(self.html, display_id=f"html{self.dispid}")
        self.negatives = defaultdict(lambda: 0)
        self.book = set()
        self.engine = Stockfish()

    def fen(self):
        return self.board.fen()

    def stockfish(self, playmove=False, time=1.0):
        self.engine.set_fen_position(fen_position = self.board.fen())
        move = self.engine.get_best_move_time(time=1.0)
        if move is None:
            return None
        move_uci = self.board.parse_uci(move)
        move_san = self.board.san(move_uci)
        if playmove:
            self.board.push_san(move_san)
            self.moves.append(move_san)
            self.game = self.game + " " + move_san if self.game != "" else move_san
            self.update()
        return move_san

    async def play(self, move_arg=None):
        """
        If a move is not given, attempt to generate one with neural net.
        Otherwise accept the move as given.
        Then play the move.
        """
        # print("play")
        def propose_move(game):
            return autocomplete(
                prompt="\n"+game,
                model=model,
                encode=dataset.encode,
                decode=dataset.decode,
                n_ctx=len(game)+13,
                n_generate=12,
                temp=1.0,
                device="cuda").split()[0]
        if move_arg is not None:
            move = self.board.san(self.board.parse_san(move_arg))
        else:
            legal = [self.board.san(move) for move in list(self.board.legal_moves)]
            # print(self.prefix() + str(legal))
            move = None
            tries = 0
            while move not in legal:
                if tries > 0:
                    break
                if self.game == "":
                    move = propose_move(self.game)
                else:
                    move = propose_move(self.game.strip() + " ")
                tries += 1
            if move not in legal:
                if self.game == "":
                    self.all_games.append(f'<span style="color:red">{move}</span>')
                else:
                    self.all_games.append(f'{self.highlight_game(self.game)} <span style="color:red">{move}</span>')
                return None
        self.board.push_san(move)
        self.moves.append(move)
        self.game = self.game + " " + move if self.game != "" else move
        self.update()
        return move

    def highlight_game(self, game):
        moves = game.split()
        hl_game = ""
        for n, move in enumerate(moves):
            if n > 0:
                hl_game += " "
            if in_book(' '.join(moves[:n])):
                hl_game += f'<span style="color:blue">{move}</span>'
            else:
                hl_game += f'{move}'
        return hl_game

    def back(self):
        self.game = ''.join(self.game.split()[:-1])
        self.board.pop()
        self.update()

    def restart(self):
        # self.all_games.append(self.game)
        self.game = ''
        for _ in range(len(self.moves)):
            self.board.pop()
        self.moves = []
        self.update()

    def update(self):
        self.all_games = self.all_games[-5:]
        self.html = HTML("<pre>" + self.highlight_game(self.game) + '\n' + '\n'.join(self.all_games[::-1]) + "</pre")
        self.html_display.update(self.html)
        self.board_display.update(self.board)
