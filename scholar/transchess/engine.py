
import numpy as np
from numba import jit #, types, typed
from numba.experimental import jitclass
from numba.types import ListType, FunctionType, int64, int64

# dict_ty = types.DictType(types.int64, types.unicode_type)


@jit(int64(int64))
def popcnt64(x):
    """
    Return the number of 1 bits in binary representation of x.
    Assumes 0 <= x < 2**64.
    """
    k1: int64 = 0x5555555555555555; k2: int64 = 0x3333333333333333;
    k4: int64 = 0x0f0f0f0f0f0f0f0f; kf: int64 = 0x0101010101010101;
    x = x - ((x >> 1) & k1)
    x = (x & k2) + ((x >> 2) & k2)
    x = (x + (x >> 4)) & k4
    x = (x * kf) >> 56
    return x

@jit(int64(int64))
def ntz64(x: int64):
    """
    Return the number of trailing zeros in the binary representation of x.
    Assumes 0 <= x < 2**64.
    """
    debruijn: int64 = 0x03f79d71b4cb0a89;
    lookup: ListType(int64) = [0, 47,  1, 56, 48, 27,  2, 60,
        57, 49, 41, 37, 28, 16,  3, 61,
        54, 58, 35, 52, 50, 42, 21, 44,
        38, 32, 29, 23, 17, 11,  4, 62,
        46, 55, 26, 59, 40, 36, 15, 53,
        34, 51, 20, 43, 31, 22, 10, 45,
        25, 39, 14, 33, 19, 30,  9, 24,
        13, 18,  8, 12,  7,  6,  5, 63]
    return lookup[(((debruijn*(x^(x-1))) >> 58) + 64)%64]

# @jit
# def bits64(x):
#     while x:
#         yield ntz64(x)
#         x &= x - 1

@jit
def bits64(x: int64):
    result = []
    while x:
        y: int64 = x & (x-1)
        result.append(x ^ y)
        x = y
    return result

((a8, b8, c8, d8, e8, f8, g8, h8),
 (a7, b7, c7, d7, e7, f7, g7, h7),
 (a6, b6, c6, d6, e6, f6, g6, h6),
 (a5, b5, c5, d5, e5, f5, g5, h5),
 (a4, b4, c4, d4, e4, f4, g4, h4),
 (a3, b3, c3, d3, e3, f3, g3, h3),
 (a2, b2, c2, d2, e2, f2, g2, h2),
 (a1, b1, c1, d1, e1, f1, g1, h1)) = (
    [[ int64(1) << (i+8*j) for i in range(8)] for j in range(8)])

file_a = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8
file_b = b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8
file_c = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8
file_d = d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8
file_e = e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8
file_f = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
file_g = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8
file_h = h1 + h2 + h3 + h4 + h5 + h6 + h7 + h8

rank_8 = a8 + b8 + c8 + d8 + e8 + f8 + g8 + h8
rank_7 = a7 + b7 + c7 + d7 + e7 + f7 + g7 + h7
rank_6 = a6 + b6 + c6 + d6 + e6 + f6 + g6 + h6
rank_5 = a5 + b5 + c5 + d5 + e5 + f5 + g5 + h5
rank_4 = a4 + b4 + c4 + d4 + e4 + f4 + g4 + h4
rank_3 = a3 + b3 + c3 + d3 + e3 + f3 + g3 + h3
rank_2 = a2 + b2 + c2 + d2 + e2 + f2 + g2 + h2
rank_1 = a1 + b1 + c1 + d1 + e1 + f1 + g1 + h1

@jit(int64(int64))
def e(x):
    return (x << 1) & ~(file_a)

@jit(int64(int64))
def w(x):
    return (x >> 1) & ~(file_h)

@jit(int64(int64))
def s(x):
    return (x << 8) & ~(rank_8)

@jit(int64(int64))
def n(x):
    return (x >> 8) & ~(rank_1)

@jit(int64(int64))
def nw(x):
    return n(w(x))

@jit(int64(int64))
def ne(x):
    return n(e(x))

@jit(int64(int64))
def sw(x):
    return s(w(x))

@jit(int64(int64))
def se(x):
    return s(e(x))

@jit(int64(int64))
def wsw(x):
    return w(sw(x))

@jit(int64(int64))
def wnw(x):
    return w(nw(x))

@jit(int64(int64))
def ene(x):
    return e(ne(x))

@jit(int64(int64))
def ese(x):
    return e(se(x))

@jit(int64(int64))
def nwn(x):
    return nw(n(x))

@jit(int64(int64))
def nen(x):
    return ne(n(x))

@jit(int64(int64))
def sws(x):
    return sw(s(x))

@jit(int64(int64))
def ses(x):
    return se(s(x))

# could we use np.bitwise_or.reduce(ar)?

@jit #(int64(int64,ListType(FunctionType(int64(int64)))))
def hopper(x: int64, S: ListType(int64(int64))) -> int64:
    bb: int64 = 0
    for f in S:
        bb |= f(x)
    return bb

@jit #(int64(int64,FunctionType(int64(int64)),int64))
def ray(x, f, empty):
    bb: int64 = 0
    y = f(x)
    while y & empty:
        bb |= y
        y = f(y)
    bb |= y
    return bb

@jit #(int64(int64,ListType(FunctionType(int64(int64))),int64))
def slider(x: int64, S, empty: int64):
    bb: int64 = 0
    for f in S:
        bb |= ray(x, f, empty)
    return bb

# move flags
standard = 1 << 1
pawnpush = 1 << 2
castleQ = 1 << 3
castleK = 1 << 4
enpassant = 1 << 5
promoteQ = 1 << 8
promoteR = 1 << 9
promoteB = 1 << 10
promoteN = 1 << 11
promote = promoteQ | promoteR | promoteB | promoteN
# double pawn pushes: the move flag will be the square in front of pawn before moving

spec = [
    ('white', int64),
    ('black', int64),
    ('king', int64),
    ('queen', int64),
    ('bishop', int64),
    ('knight', int64),
    ('rook', int64),
    ('pawn', int64),
    ('ply', int64),
    ('castling', int64),
    ('enpassant', int64),
    ('halfmove', int64),
    ('fullmove', int64)
]

@jitclass(spec)
class Engine:
    def __init__(self):
        self.white = rank_1 | rank_2
        self.black = rank_7 | rank_8
        self.king = e1 | e8
        self.queen = d1 | d8
        self.bishop = c1 | c8 | f1 | f8
        self.knight = b1 | b8 | g1 | g8
        self.rook = a1 | a8 | h1 | h8
        self.pawn = rank_2 | rank_7
        self.ply = 0
        self.castling = a1 | a8 | h1 | h8
        self.enpassant = 0
        self.halfmove = 0
        self.fullmove = 1

    def threat(self, white_to_move: bool):
        """
        A square is threatened if a pseudolegal move from the
        opposing side could target it were it unoccupied.
        """
        occupied: int64 = self.white + self.black
        empty: int64 = ~occupied
        bb: int64 = 0
        us: int64 = self.white if white_to_move else self.black

        for x in bits64(us & self.king):
            S = [n, w, s, e, nw, ne, sw, se]
            bb |= hopper(x, S)
        for x in bits64(us & (self.queen | self.rook)):
            S = [n, w, s, e]
            bb |= slider(x, S, empty)
        for x in bits64(us & (self.queen | self.bishop)):
            S = [nw, ne, sw, se]
            bb |= slider(x, S, empty)
        for x in bits64(us & self.knight):
            S = [nwn, nen, wsw, wnw, sws, ses, ese, ene]
            bb |= hopper(x, S)
        for x in bits64(self.pawn):
            S = [nw, ne] if white_to_move else [sw, se]
            bb |= hopper(x, S)
        return bb

    def pseudolegal(self):
        """
        remove the "cannot move into check" aspect from the game, and the resulting moves
        are called pseudolegal.

        note: by this definition, it is pseudolegal to castle into check, but not *through* check
        """
        moves = []
        white_to_move: bool = (self.ply % 2 == 0)
        if white_to_move:
            us: int64 = self.white
            them: int64 = self.black
            backrank: int64 = rank_1
            endrank: int64 = rank_8
            notendrank: int64 = int64(~rank_8)
            doublepawnpush: int64 = rank_4
        else:
            us: int64 = self.black
            them: int64 = self.white
            backrank: int64 = rank_8
            endrank: int64 = rank_1
            notendrank: int64 = int64(~rank_1)
            doublepawnpush: int64 = rank_5
        occupied: int64 = self.white | self.black
        empty: int64 = int64(~occupied)
        safe: int64 = int64(~self.threat(~white_to_move))
        safe_and_empty: int64 = safe & empty # the square we pass during castling must be in safe_and_empty
        empty_or_them: int64 = empty | them

        def add(source, targets, flags):
            for target in bits64(targets):
                for f in bits64(flags):
                    moves.append((source, target, f))

        for x in bits64(us & self.king):
            S = [n, w, s, e, nw, ne, sw, se]
            Y = hopper(x, S)
            add(x, Y & empty_or_them, standard)
            rooks = us & self.rook & self.castling
            add(x, e(e(e(rooks))) & e(e(empty)) & e(empty) & empty & w(safe_and_empty) & w(w(x)), castleQ)
            add(x, e(e(x)) & e(safe_and_empty) & empty & w(empty) & w(w(rooks)), castleK)

        for x in bits64(us & (self.queen | self.rook)):
            S = [n, w, s, e]
            Y = slider(x, S, empty)
            add(x, Y & empty_or_them, standard)

        for x in bits64(us & (self.queen | self.bishop)):
            S = [nw, ne, nw, se]
            Y = slider(x, S, empty)
            add(x, Y & empty_or_them, standard)

        for x in bits64(us & self.knight):
            S = [nwn, nen, wsw, wnw, sws, ses, ese, ene]
            Y = hopper(x, S)
            add(x, Y & empty_or_them, standard)

        for x in bits64(us & self.pawn):
            f = n if white_to_move else s
            add(x, f(f(x)) & f(empty) & empty & (rank_4 if white_to_move else rank_5), f(x)) # 0
            add(x, f(x) & empty & notendrank, pawnpush) # 1
            add(x, f(x) & (empty & (endrank)), promote)
            S = [nw, ne] if white_to_move else [sw, se]
            Y = hopper(x, S)
            add(x, Y & self.enpassant, enpassant)
            add(x, Y & them & notendrank, standard)
            add(x, Y & them & endrank, promote)

        return moves

    def move(self, m):
        white_to_move = (self.ply % 2 == 0)
        if white_to_move:
            (us, them) = (self.white, self.black)
            backrank = rank_1
            endrank = rank_8
        else:
            (us, them) = self.black, self.white
            backrank = rank_8
            endrank = rank_1

        (source, target, flag) = m

        if target & them != 0 or flag == enpassant:
            capture = True

        # handle target square
        # 1. if empty, no problem
        # 2. if piece, remove it from its bitboard
        # 3. remove from `them`
        # 4. if an enpassant target, capture en passant
        mask = ~target
        them &= mask
        self.queen &= mask
        self.rook &= mask
        self.bishop &= mask
        self.knight &= mask
        self.pawn &= mask
        if flag == enpassant:
            notsofastyousonofabitch = (s if white_to_move else n)(target)
            self.pawn ^= notsofastyousonofabitch
            them ^= notsofastyousonofabitch

        # handle source square
        # 1. move piece over to target square
        # 2. update castling rights if needed
        mover = source | target
        us ^= mover
        if source & self.king:
            self.king ^= mover
            if white_to_move:
                self.castling &= ~(a1 | h1)
            else:
                self.castling &= ~(a8 | h8)
        if source & self.queen:
            self.queen ^= mover
        elif source & self.rook:
            self.rook ^= mover
            self.castling &= ~source
        elif source & self.bishop:
            self.bishop ^= mover
        elif source & self.knight:
            self.knight ^= mover
        elif source & self.pawn:
            self.pawn ^= mover

        # handle promotions
        promoted = self.pawn & endrank
        self.pawn ^= promoted
        if promoted:
            if flag == promoteQ:
                self.queen ^= promoted
            elif flag == promoteR:
                self.rook ^= promoted
            elif flag == promoteB:
                self.bishop ^= promoted
            elif flag == promoteN:
                self.knight ^= promoted

        # handle castling rooks
        if flag == castleK:
            rook_mover = e(target) | w(target)
            self.rook ^= rook_mover
            us ^= rook_mover

        if flag == castleQ:
            rook_mover = w(w(target)) | e(target)
            self.rook ^= rook_mover
            us ^= rook_mover

        # handle en passant flags
        if flag & (rank_3 if white_to_move else rank_6):
            self.enpassant = flag
        else:
            self.enpassant = 0

        # handle ply
        self.ply += 1

        # handle half-move clock
        self.halfmove += 1

        if capture or (flag & (rank_3 | rank_6 | pawnpush) != 0):
            self.halfmove = 0

        # handle full-move number
        if not white_to_move:
            self.fullmove += 1

    def clone(self):
        temp = Engine()
        temp.white = self.white
        temp.black = self.black
        temp.king = self.king
        temp.queen = self.queen
        temp.bishop = self.bishop
        temp.knight = self.knight
        temp.rook = self.rook
        temp.pawn = self.pawn
        temp.ply = self.ply
        temp.castling = self.castling
        temp.enpassant = self.enpassant
        temp.halfmove = self.halfmove
        temp.fullmove = self.fullmove
        return temp

    def legal_moves(self):
        """
        A move is legal if:
          1. It is pseudolegal
          2. If performed, then there isn't a pseudolegal king capture
        """
        pseudolegal = self.pseudolegal()

        moves = []
        white_to_move = (self.ply % 2 == 0)
        if white_to_move:
            (us, them) = (self.white, self.black)
            backrank = rank_1
            endrank = rank_8
        else:
            (us, them) = self.black, self.white
            backrank = rank_8
            endrank = rank_1

        king = us & self.king

        for move in pseudolegal:
            temp = self.clone()
            temp.move(move)
            threat = temp.threat(~white_to_move)
            us_now = temp.white if white_to_move else temp.black
            if us_now & temp.king & threat == 0:
                moves.append(move)

        return moves
