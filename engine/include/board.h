// engine/include/board.h
// YichiAlpha — Board representation
//
// Compact representation of 异吃棋 board state.
// Mirrors python/game.py::GameState exactly.

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <utility>

namespace yichi {

enum class CellType : uint8_t {
    EMPTY = 0,
    X     = 1,
    O     = 2,
    BLOCK = 3,
};

struct GameConfig {
    int board_size      = 6;
    int initial_health  = 2;
    int attack_power    = 1;
    int heal_power      = 1;
    bool diag_heal      = true;
    bool diag_attack    = true;
    float block_density = 0.0f;
};

class Board {
public:
    explicit Board(const GameConfig& cfg);

    // Reset to initial empty state (with optional random blocks).
    void reset(unsigned seed = 0);

    // Deep copy.
    Board clone() const;

    // Equality (used for caching).
    bool operator==(const Board& other) const;

    // ---------- Coordinate helpers ----------
    inline int idx(int r, int c) const {
        return r * config_.board_size + c;
    }
    inline CellType at(int r, int c) const {
        return types_[idx(r, c)];
    }
    inline int8_t hp_at(int r, int c) const {
        return health_[idx(r, c)];
    }
    inline int board_size() const { return config_.board_size; }
    inline int current_player() const { return current_player_; }
    inline int step() const { return step_; }
    inline const GameConfig& config() const { return config_; }

    // ---------- Game operations ----------
    bool is_empty(int r, int c) const;
    bool is_full() const;
    std::vector<std::pair<int,int>> legal_moves() const;
    bool is_terminal() const;
    int winner() const;  // 1=X, 2=O, -1=draw, 0=ongoing
    float terminal_reward_for_current_player() const;

    // ---------- Apply move ----------
    // Place stone at (r,c), run chain refresh, switch player.
    void apply_move(int r, int c);

    // ---------- Serialization ----------
    uint64_t hash() const;
    std::string to_string() const;

    // Direct access (for rules.cpp)
    std::vector<CellType>& types_mut() { return types_; }
    std::vector<int8_t>& health_mut() { return health_; }
    const std::vector<CellType>& types() const { return types_; }
    const std::vector<int8_t>& health() const { return health_; }
    void set_current_player(int p) { current_player_ = p; }
    void set_step(int s) { step_ = s; }

private:
    GameConfig config_;
    std::vector<CellType> types_;   // size = N*N
    std::vector<int8_t>  health_;   // size = N*N
    int current_player_;            // 1=X, 2=O (matches CellType)
    int step_;
};

}  // namespace yichi
