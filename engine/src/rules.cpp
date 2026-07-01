// engine/src/rules.cpp
// YichiAlpha — Game rules implementation
//
// Strict port of python/game.py rules. Verified to produce identical results.

#include "rules.h"
#include "board.h"

#include <cassert>

namespace yichi {

// Direct (up/left/right/down) and diagonal (4 corners) neighbor offsets
constexpr int Rules::DIRECT[4][2];
constexpr int Rules::DIAGONAL[4][2];

void Rules::count_neighbors(const Board& b, int r, int c, int player,
                            int& n_same_direct, int& m_same_diag,
                            int& n_opp_direct, int& m_opp_diag) {
    const int N = b.board_size();
    const int opp = (player == 1) ? 2 : 1;
    n_same_direct = m_same_diag = n_opp_direct = m_opp_diag = 0;

    for (int i = 0; i < 4; ++i) {
        int nr = r + DIRECT[i][0];
        int nc = c + DIRECT[i][1];
        if (nr >= 0 && nr < N && nc >= 0 && nc < N) {
            int t = static_cast<int>(b.at(nr, nc));
            if (t == player) ++n_same_direct;
            else if (t == opp) ++n_opp_direct;
        }
    }
    for (int i = 0; i < 4; ++i) {
        int nr = r + DIAGONAL[i][0];
        int nc = c + DIAGONAL[i][1];
        if (nr >= 0 && nr < N && nc >= 0 && nc < N) {
            int t = static_cast<int>(b.at(nr, nc));
            if (t == player) ++m_same_diag;
            else if (t == opp) ++m_opp_diag;
        }
    }
}

void Rules::heal_rule(Board& b, int r, int c) {
    const auto& cfg = b.config();
    int me = static_cast<int>(b.at(r, c));
    if (me == 0 || me == 3) return;  // EMPTY or BLOCK

    // Reset to base health
    b.health_mut()[b.idx(r, c)] = static_cast<int8_t>(cfg.initial_health);

    int n_same_direct, m_same_diag, n_opp_direct, m_opp_diag;
    count_neighbors(b, r, c, me, n_same_direct, m_same_diag, n_opp_direct, m_opp_diag);

    if (n_same_direct >= 2) {
        if (m_same_diag > 0 && cfg.diag_heal) {
            b.health_mut()[b.idx(r, c)] =
                static_cast<int8_t>(cfg.initial_health + (n_same_direct + m_same_diag) * cfg.heal_power - 1);
        } else {
            b.health_mut()[b.idx(r, c)] =
                static_cast<int8_t>(cfg.initial_health + n_same_direct * cfg.heal_power - 1);
        }
    }
}

void Rules::damage_rule(Board& b, int r, int c, int player) {
    const auto& cfg = b.config();
    int me = static_cast<int>(b.at(r, c));
    if (me == 0 || me == 3) return;

    int n_same_direct, m_same_diag, n_opp_direct, m_opp_diag;
    count_neighbors(b, r, c, player, n_same_direct, m_same_diag, n_opp_direct, m_opp_diag);

    // Damage based on opponent neighbors
    if (n_opp_direct >= 2) {
        if (m_opp_diag > 0 && cfg.diag_attack) {
            b.health_mut()[b.idx(r, c)] -=
                static_cast<int8_t>((n_opp_direct + m_opp_diag) * cfg.attack_power);
        } else {
            b.health_mut()[b.idx(r, c)] -=
                static_cast<int8_t>(n_opp_direct * cfg.attack_power);
        }
    }
}

bool Rules::death_rule(Board& b, int r, int c) {
    int me = static_cast<int>(b.at(r, c));
    if (me == 0 || me == 3) return false;  // EMPTY or BLOCK

    if (b.hp_at(r, c) <= 0) {
        // Flip to opponent color
        b.types_mut()[b.idx(r, c)] = (me == 1) ? CellType::O : CellType::X;
        // Health resets to 2 (NOT initial_health — see game_source.js line 730)
        b.health_mut()[b.idx(r, c)] = 2;
        return true;
    }
    return false;
}

void Rules::block_rule(Board& b) {
    const int N = b.board_size();
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            if (b.at(r, c) == CellType::BLOCK) continue;
            int x_cnt = 0, o_cnt = 0;
            for (int i = 0; i < 4; ++i) {
                int nr = r + DIRECT[i][0];
                int nc = c + DIRECT[i][1];
                if (nr >= 0 && nr < N && nc >= 0 && nc < N) {
                    if (b.at(nr, nc) == CellType::X) ++x_cnt;
                    else if (b.at(nr, nc) == CellType::O) ++o_cnt;
                }
            }
            if (x_cnt >= 2 && o_cnt >= 2) {
                b.types_mut()[b.idx(r, c)] = CellType::BLOCK;
                b.health_mut()[b.idx(r, c)] = 0;
            }
        }
    }
}

bool Rules::refresh_board(Board& b) {
    const auto& cfg = b.config();
    const int N = cfg.board_size;
    bool changed = false;

    // Step A: block_rule (only if asymmetric config)
    if (!cfg.diag_heal || !cfg.diag_attack || cfg.attack_power != cfg.heal_power) {
        block_rule(b);
    }

    // Step B: heal all
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            int t = static_cast<int>(b.at(r, c));
            if (t != 0 && t != 3) heal_rule(b, r, c);
        }
    }

    // Step C: damage + death for current player
    int cur = b.current_player();
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            if (static_cast<int>(b.at(r, c)) == cur) {
                damage_rule(b, r, c, cur);
            }
        }
    }
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            if (static_cast<int>(b.at(r, c)) == cur) {
                if (death_rule(b, r, c)) changed = true;
            }
        }
    }

    // Step D: heal all again
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            int t = static_cast<int>(b.at(r, c));
            if (t != 0 && t != 3) heal_rule(b, r, c);
        }
    }

    // Step E: damage + death for opponent
    int opp = (cur == 1) ? 2 : 1;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            if (static_cast<int>(b.at(r, c)) == opp) {
                damage_rule(b, r, c, opp);
            }
        }
    }
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            if (static_cast<int>(b.at(r, c)) == opp) {
                if (death_rule(b, r, c)) changed = true;
            }
        }
    }

    return changed;
}

void Rules::apply_move(Board& b, int r, int c) {
    assert(b.is_empty(r, c) && "apply_move on non-empty cell");

    const auto& cfg = b.config();
    int p = b.current_player();

    // 1. Place stone
    b.types_mut()[b.idx(r, c)] = static_cast<CellType>(p);
    b.health_mut()[b.idx(r, c)] = static_cast<int8_t>(cfg.initial_health);

    // 2. Chain refresh until stable
    while (true) {
        bool changed = refresh_board(b);
        if (!changed) break;
    }

    // 3. Switch player
    b.set_current_player((p == 1) ? 2 : 1);
    b.set_step(b.step() + 1);
}

}  // namespace yichi
