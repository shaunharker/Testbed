// chessboard.cpp
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
#include <cmath>

#include "chessboard.hpp"

int main(int argc, char * argv []) {
  std::deque<std::string> inputs;
  std::vector<std::string> game;
  Chessboard board;
  bool valid = true;
  while (true) {

    // Read moves from stdin
    bool fresh = true;
    while (inputs.size() == 0) {
      if (fresh) {
        fresh = false;
      } else {
        return 1;
      }
      std::string user;
      std::cin >> user;
      std::istringstream ss(user);
      std::string item;
      while(std::getline(ss, item, ' ')){
        if (item == "0-1" || item == "1-0" || item == "1/2-1/2" || item == "...") {
          if (valid && game.size() > 8) {
            std::string output_style = "echo";
            if (argc > 1) {
              output_style = std::string(argv[1]);
            }
            if (output_style == "echo") {
              for (auto move : game) {
                std::cout << move << " ";
              }
              std::cout << item << "\n";
            } else
            if (output_style == "full") {
              std::cout << "{\"game\": \"";
              for (auto move : game) {
                std::cout << move << " ";
              }
              std::cout << "\", \"outcome\": \"" << item << "\", \"legal\": [";
              auto legaljson = [&](){
                std::stringstream ss;
                ss << "[";
                bool first = true;
                for (auto alt : board.legal_moves()) {
                  if(first) first=false; else ss << ", ";
                  ss << "\"" << alt << "\"";
                }
                ss << "]";
                return ss.str();
              };
              board = Chessboard();
              std::cout << legaljson();
              for (auto move : game) {
                board.move(move);
                std::cout << ", " << legaljson();
              }
              std::cout << "], \"fen\": [";
              board = Chessboard();
              std::cout << "\"" << board.fen() << "\"";
              for (auto move : game) {
                board.move(move);
                std::cout << ", \"" << board.fen() << "\"";
              }
              std::cout << "]}\n";
            }
          }

          game.clear();
          board = Chessboard();
          fresh = true;
          valid = true;
        } else {
          inputs.push_back(item);
        }
      }
    }

    std::string move = inputs.front();
    inputs.pop_front();
    game.push_back(move);
    valid &= board.move(move);
  }
  return 0;
}

/*

Test game:

d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Bg5 c5 d5 a6 a4 e6 Qd2 exd5 exd5 Re8 h3 Qa5 Nf3 b5 Bxf6 Bxf6 Ne4 Qxd2+ Nfxd2 Bxb2 Ra2 Be5 f4 Bg7 Nxd6 Rd8 Nxc8 Rxc8 axb5 a5 Ne4 Nd7 Kd2 Re8 Bd3 Bd4 Rf1 Ra7 Rf3 Kf8 Nd6 Rd8 b6 Ra6 Nb5 Bg7 b7 Rb8 Nc7 Ra7 Nb5 Ra6 Be2 Bf6 Nc7 Ra7 Nb5 Ra6 Rfa3 Bd8 d6 Rxb7 Bf3 Rb8 Kd3 Nf6 Rd2 Ne8 Kc2 Rbb6 Rad3 h6 Rd5 a4 Kb1 a3 Ka2 Ra4 Rxc5 Rb8 d7 Ke7 Re5+ Kf6 dxe8=N#

*/
