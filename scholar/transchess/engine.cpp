// engine.cpp
// need a passable chess engine
// Shaun Harker 2022-05-01
// MIT LICENSE

#include <array>
#include <vector>
#include <deque>
#include <cstdint>
#include <iostream>
#include <functional>
#include <sstream>
#include <chrono>
#include <thread>

typedef std::function<uint64_t(uint64_t)> Fun;
typedef std::array<Fun,2> FunList2;
typedef std::array<Fun,4> FunList4;
typedef std::array<Fun,8> FunList8;
typedef std::tuple<uint64_t, uint64_t, uint64_t> Move;
typedef std::vector<Move> MoveList;


// castling rights
// a1, h1 are move flags representing white castling rights 1<<56 1<<63
// a8, h8 are move flags representing black castling rights 1<<0 1<<7

// move flags
constexpr uint64_t captureflag = uint64_t(1) << 1;
constexpr uint64_t checkflag = uint64_t(1) << 2;
constexpr uint64_t mateflag = uint64_t(1) << 3;
constexpr uint64_t standard = uint64_t(1) << 4;
constexpr uint64_t pawnpush = uint64_t(1) << 5;
constexpr uint64_t castleQ = uint64_t(1) << 6;
constexpr uint64_t castleK = uint64_t(1) << 8;
constexpr uint64_t enpassantflag = uint64_t(1) << 9;
constexpr uint64_t promoteQ = uint64_t(1) << 10;
constexpr uint64_t promoteR = uint64_t(1) << 11;
constexpr uint64_t promoteB = uint64_t(1) << 12;
constexpr uint64_t promoteN = uint64_t(1) << 13;
constexpr uint64_t promote = promoteQ | promoteR | promoteB | promoteN;

int popcnt(uint64_t x) {
  // Return the number of 1 bits in binary representation of x.
  // Assumes 0 <= x < 2**64.
  // (todo: is this a processor instruction now?)
  uint64_t k1 = 0x5555555555555555; uint64_t k2 = 0x3333333333333333;
  uint64_t k4 = 0x0f0f0f0f0f0f0f0f; uint64_t kf = 0x0101010101010101;
  x = x - ((x >> 1) & k1);
  x = (x & k2) + ((x >> 2) & k2);
  x = (x + (x >> 4)) & k4;
  x = (x * kf) >> 56;
  return x;
}

int ntz(uint64_t x) {
    // We return the number of trailing zeros in
    // the binary representation of x.
    //
    // We have that 0 <= x < 2^64.
    //
    // We begin by applying a function sensitive only
    // to the least significant bit (lsb) of x:
    //
    //   x -> x^(x-1)  e.g. 0b11001000 -> 0b00001111
    //
    // Observe that x^(x-1) == 2^(ntz(x)+1) - 1.

    uint64_t y = x^(x-1);

    // Next, we multiply by 0x03f79d71b4cb0a89,
    // and then roll off the first 58 bits.

    constexpr uint64_t debruijn = 0x03f79d71b4cb0a89;

    uint8_t z = (debruijn*y) >> 58;

    // What? Don't look at me like that.
    //
    // With 58 bits rolled off, only 6 bits remain,
    // so we must have one of 0, 1, 2, ..., 63.
    //
    // It turns out this number was judiciously
    // chosen to make it so each of the possible
    // values for y were mapped into distinct slots.
    // See https://en.wikipedia.org/wiki/De_Bruijn_sequence#Finding_least-_or_most-significant_set_bit_in_a_word
    //
    // So we just use a look-up table of all 64
    // possible answers, which have been precomputed in
    // advance by the the sort of people who write
    // chess engines in their spare time:

    constexpr std::array<int,64> lookup = {
         0, 47,  1, 56, 48, 27,  2, 60,
        57, 49, 41, 37, 28, 16,  3, 61,
        54, 58, 35, 52, 50, 42, 21, 44,
        38, 32, 29, 23, 17, 11,  4, 62,
        46, 55, 26, 59, 40, 36, 15, 53,
        34, 51, 20, 43, 31, 22, 10, 45,
        25, 39, 14, 33, 19, 30,  9, 24,
        13, 18,  8, 12,  7,  6,  5, 63
    };

    return lookup[z];
}


std::string square(int s) {
  char cstr[3] {(char)('a' + s%8), (char)('8' - s/8), 0};
  return std::string(cstr);
}

// double pawn pushes: the move flag will be the square in front of pawn before moving


// ((a8, b8, c8, d8, e8, f8, g8, h8),
//  (a7, b7, c7, d7, e7, f7, g7, h7),
//  (a6, b6, c6, d6, e6, f6, g6, h6),
//  (a5, b5, c5, d5, e5, f5, g5, h5),
//  (a4, b4, c4, d4, e4, f4, g4, h4),
//  (a3, b3, c3, d3, e3, f3, g3, h3),
//  (a2, b2, c2, d2, e2, f2, g2, h2),
//  (a1, b1, c1, d1, e1, f1, g1, h1)) = (
//     [[ uint64_t64(1) << (i+8*j) for i in range(8)] for j in range(8)])
//


constexpr uint64_t a8 = uint64_t(1) << 0;
constexpr uint64_t b8 = uint64_t(1) << 1;
constexpr uint64_t c8 = uint64_t(1) << 2;
constexpr uint64_t d8 = uint64_t(1) << 3;
constexpr uint64_t e8 = uint64_t(1) << 4;
constexpr uint64_t f8 = uint64_t(1) << 5;
constexpr uint64_t g8 = uint64_t(1) << 6;
constexpr uint64_t h8 = uint64_t(1) << 7;

constexpr uint64_t a7 = uint64_t(1) << 8;
constexpr uint64_t b7 = uint64_t(1) << 9;
constexpr uint64_t c7 = uint64_t(1) << 10;
constexpr uint64_t d7 = uint64_t(1) << 11;
constexpr uint64_t e7 = uint64_t(1) << 12;
constexpr uint64_t f7 = uint64_t(1) << 13;
constexpr uint64_t g7 = uint64_t(1) << 14;
constexpr uint64_t h7 = uint64_t(1) << 15;

constexpr uint64_t a6 = uint64_t(1) << 16;
constexpr uint64_t b6 = uint64_t(1) << 17;
constexpr uint64_t c6 = uint64_t(1) << 18;
constexpr uint64_t d6 = uint64_t(1) << 19;
constexpr uint64_t e6 = uint64_t(1) << 20;
constexpr uint64_t f6 = uint64_t(1) << 21;
constexpr uint64_t g6 = uint64_t(1) << 22;
constexpr uint64_t h6 = uint64_t(1) << 23;

constexpr uint64_t a5 = uint64_t(1) << 24;
constexpr uint64_t b5 = uint64_t(1) << 25;
constexpr uint64_t c5 = uint64_t(1) << 26;
constexpr uint64_t d5 = uint64_t(1) << 27;
constexpr uint64_t e5 = uint64_t(1) << 28;
constexpr uint64_t f5 = uint64_t(1) << 29;
constexpr uint64_t g5 = uint64_t(1) << 30;
constexpr uint64_t h5 = uint64_t(1) << 31;

constexpr uint64_t a4 = uint64_t(1) << 32;
constexpr uint64_t b4 = uint64_t(1) << 33;
constexpr uint64_t c4 = uint64_t(1) << 34;
constexpr uint64_t d4 = uint64_t(1) << 35;
constexpr uint64_t e4 = uint64_t(1) << 36;
constexpr uint64_t f4 = uint64_t(1) << 37;
constexpr uint64_t g4 = uint64_t(1) << 38;
constexpr uint64_t h4 = uint64_t(1) << 39;

constexpr uint64_t a3 = uint64_t(1) << 40;
constexpr uint64_t b3 = uint64_t(1) << 41;
constexpr uint64_t c3 = uint64_t(1) << 42;
constexpr uint64_t d3 = uint64_t(1) << 43;
constexpr uint64_t e3 = uint64_t(1) << 44;
constexpr uint64_t f3 = uint64_t(1) << 45;
constexpr uint64_t g3 = uint64_t(1) << 46;
constexpr uint64_t h3 = uint64_t(1) << 47;

constexpr uint64_t a2 = uint64_t(1) << 48;
constexpr uint64_t b2 = uint64_t(1) << 49;
constexpr uint64_t c2 = uint64_t(1) << 50;
constexpr uint64_t d2 = uint64_t(1) << 51;
constexpr uint64_t e2 = uint64_t(1) << 52;
constexpr uint64_t f2 = uint64_t(1) << 53;
constexpr uint64_t g2 = uint64_t(1) << 54;
constexpr uint64_t h2 = uint64_t(1) << 55;

constexpr uint64_t a1 = uint64_t(1) << 56;
constexpr uint64_t b1 = uint64_t(1) << 57;
constexpr uint64_t c1 = uint64_t(1) << 58;
constexpr uint64_t d1 = uint64_t(1) << 59;
constexpr uint64_t e1 = uint64_t(1) << 60;
constexpr uint64_t f1 = uint64_t(1) << 61;
constexpr uint64_t g1 = uint64_t(1) << 62;
constexpr uint64_t h1 = uint64_t(1) << 63;

constexpr uint64_t file_a = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
constexpr uint64_t file_b = b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8;
constexpr uint64_t file_c = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8;
constexpr uint64_t file_d = d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8;
constexpr uint64_t file_e = e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8;
constexpr uint64_t file_f = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
constexpr uint64_t file_g = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8;
constexpr uint64_t file_h = h1 + h2 + h3 + h4 + h5 + h6 + h7 + h8;

constexpr uint64_t rank_8 = a8 + b8 + c8 + d8 + e8 + f8 + g8 + h8;
constexpr uint64_t rank_7 = a7 + b7 + c7 + d7 + e7 + f7 + g7 + h7;
constexpr uint64_t rank_6 = a6 + b6 + c6 + d6 + e6 + f6 + g6 + h6;
constexpr uint64_t rank_5 = a5 + b5 + c5 + d5 + e5 + f5 + g5 + h5;
constexpr uint64_t rank_4 = a4 + b4 + c4 + d4 + e4 + f4 + g4 + h4;
constexpr uint64_t rank_3 = a3 + b3 + c3 + d3 + e3 + f3 + g3 + h3;
constexpr uint64_t rank_2 = a2 + b2 + c2 + d2 + e2 + f2 + g2 + h2;
constexpr uint64_t rank_1 = a1 + b1 + c1 + d1 + e1 + f1 + g1 + h1;

auto w(uint64_t x) -> uint64_t {return (x >> 1) & ~(file_h);}
auto e(uint64_t x) -> uint64_t {return (x << 1) & ~(file_a);}
auto s(uint64_t x) -> uint64_t {return (x << 8) & ~(rank_8);}
auto n(uint64_t x) -> uint64_t {return (x >> 8) & ~(rank_1);}
auto nw(uint64_t x) -> uint64_t {return n(w(x));}
auto ne(uint64_t x) -> uint64_t {return n(e(x));}
auto sw(uint64_t x) -> uint64_t {return s(w(x));}
auto se(uint64_t x) -> uint64_t {return s(e(x));}
auto nwn(uint64_t x) -> uint64_t {return n(w(n(x)));}
auto nen(uint64_t x) -> uint64_t {return n(e(n(x)));}
auto sws(uint64_t x) -> uint64_t {return s(w(s(x)));}
auto ses(uint64_t x) -> uint64_t {return s(e(s(x)));}
auto wnw(uint64_t x) -> uint64_t {return w(n(w(x)));}
auto ene(uint64_t x) -> uint64_t {return e(n(e(x)));}
auto wsw(uint64_t x) -> uint64_t {return w(s(w(x)));}
auto ese(uint64_t x) -> uint64_t {return e(s(e(x)));}

const FunList8 kingmoves {n, w, s, e, nw, ne, sw, se};
const FunList4 bishopmoves {nw, ne, sw, se};
const FunList4 rookmoves {n, w, s, e};
const FunList8 knightmoves {nwn, nen, wsw, wnw, sws, ses, ese, ene};
const FunList2 whitepawncaptures {nw, ne};
const FunList2 blackpawncaptures {sw, se};

template <typename Fs>
auto hopper(uint64_t x, Fs S) -> uint64_t {
  uint64_t bb = 0;
  for (auto f : S) {
    bb |= f(x);
  }
  return bb;
}

template <typename F>
auto ray(uint64_t x, F f, uint64_t empty) -> uint64_t {
  uint64_t bb = 0;
  auto y = f(x);
  while (y & empty) {
    bb |= y;
    y = f(y);
  }
  bb |= y;
  return bb;
}

template <typename Fs>
auto slider(uint64_t x, Fs S, uint64_t empty) -> uint64_t {
  uint64_t bb = 0;
  for (auto it = S.begin(); it != S.end(); ++ it) {
    auto f = *it;
    bb |= ray(x, f, empty);
  }
  return bb;
}

template <typename F>
void bitapply(uint64_t x, F f) {
  uint64_t tmp, lsb;
  while (x) {
    tmp = x & (x-1);
    lsb = x ^ tmp;
    f(lsb);
    x = tmp;
  }
}

class Engine {
public:
  uint64_t white;
  uint64_t black;
  uint64_t king;
  uint64_t queen;
  uint64_t rook;
  uint64_t bishop;
  uint64_t knight;
  uint64_t pawn;
  uint64_t ply;
  uint64_t castling;
  uint64_t enpassant;
  uint64_t halfmove;
  uint64_t fullmove;
  mutable bool cache_is_computed;
  mutable std::vector<Move> _moves;
  mutable std::vector<std::string> _sanmoves;
  Engine(){
    white = rank_1 | rank_2;
    black = rank_7 | rank_8;
    king = e1 | e8;
    queen = d1 | d8;
    rook = a1 | a8 | h1 | h8;
    bishop = c1 | c8 | f1 | f8;
    knight = b1 | b8 | g1 | g8;
    pawn = rank_2 | rank_7;
    ply = 0;
    castling = a1 | a8 | h1 | h8;
    enpassant = 0;
    halfmove = 0;
    fullmove = 1;
    cache_is_computed = false;
  }

  uint64_t checked(uint64_t them) const {
    // return bitboard of attacks due to pieces on the bitboard `them`
    uint64_t bb = 0;
    uint64_t empty = ~(white | black);
    bitapply(them & king, [&](uint64_t x) {bb |= hopper(x, kingmoves);});
    bitapply(them & (queen | rook), [&](uint64_t x) {
      bb |= slider(x, rookmoves, empty);});
    bitapply(them & (queen | bishop), [&](uint64_t x) {
      bb |= slider(x, bishopmoves, empty);});
    bitapply(them & knight, [&](uint64_t x) {bb |= hopper(x, knightmoves);});
    auto pawncaptures = (them & white) ? whitepawncaptures : blackpawncaptures;
    bitapply(them & pawn, [&](uint64_t x) {bb |= hopper(x, pawncaptures);});
    return bb;
  }

  MoveList pseudolegal() const {
    // Remove the "cannot move into check" rule from the game,
    // and the resulting moves are called pseudolegal moves.
    //
    // However we distinguish this from "cannot castle through check"
    // which we regard as an independent rule, as it is based on the
    // enemy threat before the move rather than after the move.
    //
    // Therefore, it is pseudolegal to castle into check as long
    // as it is not *through* check.
    //
    // I might change this, though. Ha!
    //
    // Another comment: capture flags and check flags remain off
    //                  during this stage or else there is trouble
    //                  in the way we create pawn promotions
    MoveList pseudomoves;
    bool white_to_move = (ply % 2 == 0);
    uint64_t us = white_to_move ? white : black;
    uint64_t them = white_to_move ? black : white;
    uint64_t backrank = white_to_move ? rank_1 : rank_8;
    uint64_t endrank = white_to_move ? rank_8 : rank_1;
    uint64_t notendrank = white_to_move ? ~rank_8 : ~rank_1;
    uint64_t doublepawnpush = white_to_move ? rank_4 : rank_5;
    uint64_t occupied = white | black;
    uint64_t empty = ~occupied;
    uint64_t safe = ~checked(them);
    uint64_t safe_and_empty = safe & empty; // the square we pass during castling must be in safe_and_empty
    uint64_t empty_or_them = empty | them;

    auto add = [&](uint64_t source, uint64_t targets, uint64_t flags) {
      bitapply(targets, [&](uint64_t target){
        bitapply(flags, [&](uint64_t flag) {
          pseudomoves.push_back({source, target, flag});
        });
      });
    };

    bitapply(us & king, [&](uint64_t x){
      auto Y = hopper(x, kingmoves);
      add(x, Y & empty_or_them, standard);
      uint64_t castlerooks = us & rook & castling;
      add(x, e(e(castlerooks)) & e(empty) & empty & w(safe_and_empty) & w(w(x)), castleQ);
      add(x, e(e(x)) & e(safe_and_empty) & empty & w(castlerooks), castleK);
    });

    bitapply(us & (queen | rook), [&](uint64_t x){
      auto Y = slider(x, rookmoves, empty);
      add(x, Y & empty_or_them, standard);
    });

    bitapply(us & (queen | bishop), [&](uint64_t x){
      auto Y = slider(x, bishopmoves, empty);
      add(x, Y & empty_or_them, standard);
    });

    bitapply(us & knight, [&](uint64_t x){
      auto Y = hopper(x, knightmoves);
      add(x, Y & empty_or_them, standard);
    });

    auto f = white_to_move ? n : s;
    auto pawncaptures = white_to_move ? whitepawncaptures : blackpawncaptures;
    bitapply(us & pawn, [&](uint64_t x){
      add(x, f(f(x)) & f(empty) & empty & doublepawnpush, f(x));
      add(x, f(x) & empty & notendrank, pawnpush);
      add(x, f(x) & empty & endrank, promote);
      auto Y = hopper(x, pawncaptures);
      add(x, Y & enpassant, enpassantflag);
      add(x, Y & them & notendrank, standard);
      add(x, Y & them & endrank, promote);
    });

    return pseudomoves;
  }

  bool move(std::string s) {
    // linear scan to find move (since list is discarded
    // afterwards):
    if (!cache_is_computed) legal_moves();
    int i=0;
    for (; i<_sanmoves.size(); ++i) {
      if (s == _sanmoves[i]) {
        _move(_moves[i]);
        return true;
      }
    }
    if (i == _sanmoves.size()) {
      std::cerr << "move: '" << s << "' is not a legal move.\n";
    }
    return false;
  }

  void _move(Move m) {
    bool white_to_move = (ply % 2 == 0);
    uint64_t us = white_to_move ? white : black;
    uint64_t them = white_to_move ? black : white;
    uint64_t backrank = white_to_move ? rank_1 : rank_8;
    uint64_t endrank = white_to_move ? rank_8 : rank_1;
    const auto& [source, target, flag] = m;

    // handle unset flags
    // 1. assume promotion is queen if applicable
    // 2. detect possible en passant
    // 3. detect possible kingside castle
    // 4. detect possible queenside castle
    // 5. set flags properly

    // todo

    // handle target square
    // 1. if empty, no problem
    // 2. if piece, remove it from its bitboard
    // 3. remove from `them`
    // 4. if an enpassant target, capture en passant

    uint64_t mask = ~target;
    them &= mask;
    queen &= mask;
    rook &= mask;
    bishop &= mask;
    knight &= mask;
    pawn &= mask;
    if (flag & enpassantflag) {
      // en passant capture. "notsofast" is 1 << i,
      // where i in {0, ..., 63} is the square the
      // pawn being captured occupies.
      uint64_t notsofast = (white_to_move ? s : n)(target);
      pawn ^= notsofast;
      them ^= notsofast;
    }

    // handle source square
    // 1. move piece over to target square
    // 2. update castling rights if needed
    uint64_t mover = source | target;
    us ^= mover;
    if (source & king) {
      king ^= mover;
      if (white_to_move) {
        castling &= ~(a1 | h1);
      } else {
        castling &= ~(a8 | h8);
      }
    } else if (source & queen) {
      queen ^= mover;
    } else if (source & rook) {
      rook ^= mover;
      castling &= ~source;
    } else if (source & bishop) {
      bishop ^= mover;
    } else if (source & knight) {
      knight ^= mover;
    } else if (source & pawn) {
      pawn ^= mover;
    }

    // handle promotions
    uint64_t promoted = pawn & endrank;
    pawn ^= promoted;

    if (promoted) {
      if (flag & promoteQ) {
        queen ^= promoted;
      } else if (flag & promoteR) {
        rook ^= promoted;
      } else if (flag & promoteB) {
        bishop ^= promoted;
      } else if (flag & promoteN) {
        knight ^= promoted;
      }
    }

    // handle castling rooks
    if (flag & castleK) {
        uint64_t rook_mover = e(target) | w(target);
        rook ^= rook_mover;
        us ^= rook_mover;
    }

    if (flag & castleQ) {
        uint64_t rook_mover = w(w(target)) | e(target);
        rook ^= rook_mover;
        us ^= rook_mover;
    }

    // handle en passant
    if (flag & (white_to_move ? rank_3 : rank_6)) {
        enpassant = flag; // flag is on square behind double pushed pawn
    } else {
        enpassant = 0;
    }

    // handle ply
    ply += 1;

    // handle half-move clock
    halfmove += 1;
    if ((target & them != 0) ||
        (flag & enpassant) ||
        ((flag & (rank_3 | rank_6 | pawnpush)) != 0)) {
      halfmove = 0;
    }

    // handle full-move number
    if (!white_to_move) {
      fullmove += 1;
    }

    // set white and black
    white = white_to_move ? us : them;
    black = white_to_move ? them : us;

    // clear the cache
    cache_is_computed = false;
    _moves.clear();
    _sanmoves.clear();
  }

  MoveList _legal_moves(bool only_mate_check=false) const {
    // A move is legal if:
    //   1. It is pseudolegal
    //   2. If performed, then there isn't a pseudolegal king capture

    // Note:
    // The "check_for_mates" feature is used like this:
    // the function calls itself with it set to false
    // so that it doesn't infinitely recurse trying to
    // evaluate the legal most list of the mated position.

    bool white_to_move = (ply % 2 == 0);
    auto us = white_to_move ? white : black;
    auto them = white_to_move ? black : white;
    auto backrank = white_to_move ? rank_1 : rank_8;
    auto endrank = white_to_move ? rank_8 : rank_1;
    MoveList moves;
    MoveList pl_moves = pseudolegal();
    for (auto [source, target, flag] : pl_moves) {
      // We filter out moves that leave us in check and also
      // add flags for whether a legal move is a check or a mate.
      Engine after(*this); // fork a new position
      after._move({source, target, flag});
      uint64_t us_after = white_to_move ? after.white : after.black;
      uint64_t them_after = white_to_move ? after.black : after.white;
      if (popcnt(them_after) < popcnt(them)) flag |= captureflag;
      bool moved_into_check = (us_after & after.king & after.checked(them_after)) != 0;
      bool move_is_a_check = (them_after & after.king & after.checked(us_after)) != 0;
      if (!moved_into_check) {
        if (move_is_a_check) {
          flag |= checkflag;
          if (!only_mate_check && (after._legal_moves(true).size() == 0)) {
            flag |= mateflag;
          }
        }
        moves.push_back({source, target, flag});
        if (only_mate_check) {
          return moves;
        }
      }
    }
    _moves = moves;
    return moves;
  }

  std::vector<std::string> legal_moves() {
    if (cache_is_computed) {
      return _sanmoves;
    }
    _moves = _legal_moves(false);
    std::vector<std::string> almost_sanmoves;
    for (const auto& [source, target, flag] : _moves) {
      std::ostringstream ss;
      if (flag & castleQ) {
        ss << "O-O-O";
      } else if (flag & castleK) {
        ss << "O-O";
      } else {
        bool capture = flag & captureflag;
        if (source & king) {
          ss << "K";
        } else if (source & queen) {
          ss << "Q";
        } else if (source & bishop) {
          ss << "B";
        } else if (source & knight) {
          ss << "N";
        } else if (source & rook) {
          ss << "R";
        }
        // disambiguation
        ss << square(ntz(source));
        if (capture) {
          ss << "x";
        }
        ss << square(ntz(target));
        if (flag & promoteQ) {
          ss << "=Q";
        }
        if (flag & promoteR) {
          ss << "=R";
        }
        if (flag & promoteB) {
          ss << "=B";
        }
        if (flag & promoteN) {
          ss << "=N";
        }
      }
      if (flag & checkflag) {
        if (flag & mateflag) {
          ss << "#";
        } else {
          ss << "+";
        }
      }
      almost_sanmoves.push_back(ss.str());
    }

    // remove unnecessary disambiguation
    for (auto move1 : almost_sanmoves) {
      std::ostringstream ss;
      if (move1[0] == 'O') {
        ss << move1;
      } else if (move1[0] == 'K') {
        ss << 'K' << move1.substr(3, std::string::npos);
      } else if (move1[0] == 'Q' || move1[0] == 'R' || move1[0] == 'B' || move1[0] == 'N') {
        bool piece_ambiguity = false;
        bool file_ambiguity = false;
        bool rank_ambiguity = false;
        for (auto move2 : almost_sanmoves) {
          // different source?
          if (move1.substr(1,2) == move2.substr(1,2)) continue;
          // same piece type?
          if (move1[0] != move2[0]) continue;
          // same target square?
          if (move1[3] == 'x') {
            if (move2[3] != 'x') continue;
            if ((move1[4] != move2[4]) || (move1[5] != move2[5])) {
              continue;
            }
          } else {
            if ((move1[3] != move2[3]) || (move1[4] != move2[4])) {
              continue;
            }
          }
          piece_ambiguity = true;
          // can we not disambiguate by file?
          if ((move1[1] == move2[1]) && (move1[2] != move2[2])) {
            file_ambiguity = true;
          }
          // can we not disambiguate by rank?
          if ((move1[1] != move2[1]) && (move1[2] == move2[2])) {
            rank_ambiguity = true;
          }
        }
        ss << move1[0];
        if (piece_ambiguity) {
          if (file_ambiguity) {
            if (rank_ambiguity) {
              ss << move1.substr(1, std::string::npos);
            } else {
              ss << move1.substr(2, std::string::npos);
            }
          } else {
            ss << move1[1] << move1.substr(3, std::string::npos);
          }
        } else {
          ss << move1.substr(3, std::string::npos);
        }
      } else {
        // pawns
        if (move1[2] == 'x') {
          ss << move1[0] << move1.substr(2, std::string::npos);
        } else {
          ss << move1.substr(2, std::string::npos);
        }
      }
      _sanmoves.push_back(ss.str());
    }
    cache_is_computed = true;
    return _sanmoves;
  }

  std::string board() {
    // display board
    uint64_t s = 1;
    std::ostringstream ss;
    for (int i=0; i<8; ++i) {
      for (int j=0; j<8; ++j) {
        if (s & white) {
          if (s & king) {
            ss << "K";
          } else if (s & queen) {
            ss << "Q";
          } else if (s & bishop) {
            ss << "B";
          } else if (s & knight) {
            ss << "N";
          } else if (s & rook) {
            ss << "R";
          } else if (s & pawn) {
            ss << "P";
          } else {
            ss << "?"; // debug
          }
        } else if (s & black) {
          if (s & king) {
            ss << "k";
          } else if (s & queen) {
            ss << "q";
          } else if (s & bishop) {
            ss << "b";
          } else if (s & knight) {
            ss << "n";
          } else if (s & rook) {
            ss << "r";
          } else if (s & pawn) {
            ss << "p";
          }
        } else {
          ss << ".";
        }
        s <<= 1;
      }
      ss << "\n";
    }
    ss << "\n";
    return ss.str();
  }

  std::string fen() {
    // display board
    uint64_t s = 1;
    std::ostringstream ss;
    for (int i=0; i<8; ++i) {
      for (int j=0; j<8; ++j) {
        if (s & white) {
          if (s & king) {
            ss << "K";
          } else if (s & queen) {
            ss << "Q";
          } else if (s & bishop) {
            ss << "B";
          } else if (s & knight) {
            ss << "N";
          } else if (s & rook) {
            ss << "R";
          } else if (s & pawn) {
            ss << "P";
          } else {
            ss << "?"; // debug
          }
        } else if (s & black) {
          if (s & king) {
            ss << "k";
          } else if (s & queen) {
            ss << "q";
          } else if (s & bishop) {
            ss << "b";
          } else if (s & knight) {
            ss << "n";
          } else if (s & rook) {
            ss << "r";
          } else if (s & pawn) {
            ss << "p";
          }
        } else {
          ss << ".";
        }
        s <<= 1;
      }
      ss << "\n";
    }
    return ss.str();
  }

};

int main(int argc, char * argv []) {
  bool uci_mode = false;
  std::deque<std::string> inputs;
  std::vector<std::string> game;
  Engine engine;
  while (true) {
    // display board
    std::cout << "Board:\n\n" << engine.board() << "\n";
    // display game
    std::cout << "Game: ";
    for (auto move : game) {
      std::cout << move << " ";
    }
    std::cout << "\b\n\n";

    // display legal moves
    std::cout << "Legal Moves: ";
    std::vector<std::string> moves = engine.legal_moves();
    for (auto move : moves) std::cout << move << " ";
    std::cout << "\b\n";

    // Read moves from stdin
    bool first = true;
    while (inputs.size() == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      if (first) {
        std::cout << "> ";
        first = false;
      } else {
        return 0;
      }
      std::string user;
      std::cin >> user;
      std::istringstream ss(user);
      std::string item;
      while(std::getline(ss, item, ' ')){
        if (item == "new") {
          game.clear();
          engine = Engine();
        } else {
          inputs.push_back(item);
        }
      }
    }

    //if (inputs.size() > 0) {
    std::string move = inputs.front();
    inputs.pop_front();
    if ( move == "" ) {
      std::cout << "Unexpected doom.\n";
      return 0;
    } else if (engine.move(move)) {
      game.push_back(move);
    } else {
      // rejected move, probably a new game or bad data.
      // goal is to skip to next recognizable game.
      if (game.size()) {
        game.clear();
        inputs.push_front(move); // give it one more shot with new engine in case its the start of a new game.
      }
      engine = Engine();
    }
    //}
  }
  return 0;
}

/*

Test game:

d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Bg5 c5 d5 a6 a4 e6 Qd2 exd5 exd5 Re8 h3 Qa5 Nf3 b5 Bxf6 Bxf6 Ne4 Qxd2+ Nfxd2 Bxb2 Ra2 Be5 f4 Bg7 Nxd6 Rd8 Nxc8 Rxc8 axb5 a5 Ne4 Nd7 Kd2 Re8 Bd3 Bd4 Rf1 Ra7 Rf3 Kf8 Nd6 Rd8 b6 Ra6 Nb5 Bg7 b7 Rb8 Nc7 Ra7 Nb5 Ra6 Be2 Bf6 Nc7 Ra7 Nb5 Ra6 Rfa3 Bd8 d6 Rxb7 Bf3 Rb8 Kd3 Nf6 Rd2 Ne8 Kc2 Rbb6 Rad3 h6 Rd5 a4 Kb1 a3 Ka2 Ra4 Rxc5 Rb8 d7 Ke7 Re5+ Kf6 dxe8=N#

*/
