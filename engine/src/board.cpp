// engine/src/board.cpp
// YichiAlpha — Board implementation

#include "board.h"
#include "rules.h"   // for Rules::apply_move delegation

#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <random>
#include <chrono>

namespace yichi {

Board::Board(const GameConfig& cfg)
    : config_(cfg),
      types_(cfg.board_size * cfg.board_size, CellType::EMPTY),
      health_(cfg.board_size * cfg.board_size, 0),
      current_player_(static_cast<int>(CellType::X)),  // X = 1
      step_(0) {}

void Board::reset(unsigned seed) {
    std::fill(types_.begin(), types_.end(), CellType::EMPTY);
    std::fill(health_.begin(), health_.end(), 0);
    current_player_ = static_cast<int>(CellType::X);
    step_ = 0;

    if (config_.block_density > 0.0f) {
        // Randomly scatter blocks
        std::mt19937 rng(seed ? seed :
                         std::chrono::steady_clock::now().time_since_epoch().count());
        const int N = config_.board_size;
        const int n_blocks = static_cast<int>(std::round(N * N * config_.block_density));
        std::uniform_int_distribution<int> dist(0, N * N - 1);
        int placed = 0;
        while (placed < n_blocks && placed < N * N - 1) {
            int i = dist(rng);
            if (types_[i] == CellType::EMPTY) {
                types_[i] = CellType::BLOCK;
                health_[i] = 0;
                ++placed;
            }
        }
    }
}

Board Board::clone() const {
    Board b(config_);
    b.types_ = types_;
    b.health_ = health_;
    b.current_player_ = current_player_;
    b.step_ = step_;
    return b;
}

bool Board::operator==(const Board& other) const {
    return config_.board_size == other.config_.board_size
        && types_ == other.types_
        && health_ == other.health_
        && current_player_ == other.current_player_
        && step_ == other.step_;
}

// ---------- Game operations ----------
bool Board::is_empty(int r, int c) const {
    return types_[idx(r, c)] == CellType::EMPTY;
}

bool Board::is_full() const {
    for (auto t : types_) {
        if (t == CellType::EMPTY) return false;
    }
    return true;
}

std::vector<std::pair<int,int>> Board::legal_moves() const {
    const int N = config_.board_size;
    std::vector<std::pair<int,int>> result;

    // Check if any non-empty cell exists
    bool has_pieces = false;
    for (auto t : types_) {
        if (t != CellType::EMPTY) { has_pieces = true; break; }
    }
    if (!has_pieces) {
        // Opening: all empty cells are legal
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c)
                if (types_[idx(r,c)] == CellType::EMPTY)
                    result.emplace_back(r, c);
        return result;
    }

    // Otherwise: empty cells within 2-Chebyshev distance of any non-empty cell
    std::vector<bool> mask(N * N, false);
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            if (types_[idx(r,c)] == CellType::EMPTY) continue;
            for (int dr = -2; dr <= 2; ++dr) {
                for (int dc = -2; dc <= 2; ++dc) {
                    int nr = r + dr, nc = c + dc;
                    if (nr >= 0 && nr < N && nc >= 0 && nc < N) {
                        mask[nr * N + nc] = true;
                    }
                }
            }
        }
    }
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            if (mask[idx(r,c)] && types_[idx(r,c)] == CellType::EMPTY)
                result.emplace_back(r, c);
    return result;
}

bool Board::is_terminal() const {
    if (is_full()) return true;
    if (legal_moves().empty()) return true;
    return false;
}

int Board::winner() const {
    int x_count = 0, o_count = 0;
    for (auto t : types_) {
        if (t == CellType::X) ++x_count;
        else if (t == CellType::O) ++o_count;
    }
    if (x_count > o_count) return static_cast<int>(CellType::X);
    if (o_count > x_count) return static_cast<int>(CellType::O);
    return -1;
}

float Board::terminal_reward_for_current_player() const {
    if (!is_terminal()) return 0.0f;
    int w = winner();
    if (w == -1) return 0.0f;
    return (w == current_player_) ? 1.0f : -1.0f;
}

// ---------- Apply move ----------
// Delegates to Rules::apply_move which contains the chain refresh logic.
void Board::apply_move(int r, int c) {
    Rules::apply_move(*this, r, c);
}

// ---------- Hashing ----------
uint64_t Board::hash() const {
    // Simple FNV-1a over the byte representation
    uint64_t h = 1469598103934665603ULL;
    for (auto t : types_) {
        h ^= static_cast<uint8_t>(t);
        h *= 1099511628211ULL;
    }
    for (auto v : health_) {
        h ^= static_cast<uint8_t>(v);
        h *= 1099511628211ULL;
    }
    h ^= static_cast<uint64_t>(current_player_);
    h *= 1099511628211ULL;
    h ^= static_cast<uint64_t>(step_);
    h *= 1099511628211ULL;
    return h;
}

std::string Board::to_string() const {
    const int N = config_.board_size;
    std::ostringstream oss;
    // Header
    oss << "   ";
    for (int c = 0; c < N; ++c) oss << (char)('A' + c) << ' ';
    oss << '\n';
    for (int r = 0; r < N; ++r) {
        if (r + 1 < 10) oss << ' ';
        oss << (r + 1) << ' ';
        for (int c = 0; c < N; ++c) {
            char ch;
            switch (at(r, c)) {
                case CellType::EMPTY: ch = '.'; break;
                case CellType::X:     ch = 'x'; break;
                case CellType::O:     ch = 'o'; break;
                case CellType::BLOCK: ch = '#'; break;
            }
            oss << ch;
            if (at(r,c) == CellType::X || at(r,c) == CellType::O) {
                oss << (int)hp_at(r, c);
            } else {
                oss << ' ';
            }
            oss << ' ';
        }
        oss << '\n';
    }
    oss << "Current: " << (current_player_ == 1 ? "X" : "O")
        << ", step: " << step_;
    return oss.str();
}

}  // namespace yichi
