// engine/include/rules.h
// YichiAlpha — Game rules (healRule / damageRule / deathRule / blockRule / refreshBoard)
//
// Strict port of python/game.py::_refresh_board, _heal_rule, _damage_rule,
// _death_rule, _block_rule.

#pragma once

#include "board.h"

namespace yichi {

class Rules {
public:
    // Apply a move + chain reaction until stable.
    // Equivalent to python/game.py::apply_move.
    static void apply_move(Board& board, int r, int c);

    // Individual rules (public for unit testing).
    static void heal_rule(Board& board, int r, int c);
    static void damage_rule(Board& board, int r, int c, int player);
    static bool death_rule(Board& board, int r, int c);
    static void block_rule(Board& board);

    // One pass of refresh_board.
    // Returns true if any death_rule flipped a piece (chain continues).
    static bool refresh_board(Board& board);

private:
    static constexpr int DIRECT[4][2] = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
    static constexpr int DIAGONAL[4][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

    // Count same/opposite color neighbors of cell (r,c) for player `player`.
    // `player` is 1 (X) or 2 (O). Returns counts via reference.
    static void count_neighbors(const Board& b, int r, int c, int player,
                                int& n_same_direct, int& m_same_diag,
                                int& n_opp_direct, int& m_opp_diag);
};

}  // namespace yichi
