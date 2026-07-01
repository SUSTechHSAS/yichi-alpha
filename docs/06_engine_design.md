# 06 · C++ 引擎设计 (C++ Engine Design)

> C++ 引擎作为生产部署用，负责高性能自对弈数据生成。
>
> Python 训练 pipeline 仍负责梯度更新（PyTorch Python API 更灵活），C++ 通过 LibTorch 加载训练好的 `.pt` 权重做推理。

## 1. 模块划分

```
engine/
├── CMakeLists.txt
├── include/
│   ├── board.h          # 棋盘表示与基本操作
│   ├── rules.h          # 规则：healRule/damageRule/deathRule/blockRule/refreshBoard
│   ├── mcts.h           # MCTS 树与搜索
│   ├── neuralnet.h      # LibTorch 网络封装
│   └── selfplay.h       # 自对弈驱动
└── src/
    ├── board.cpp
    ├── rules.cpp
    ├── mcts.cpp
    ├── neuralnet.cpp
    └── selfplay_main.cpp
```

## 2. 棋盘表示 `board.h`

```cpp
#pragma once
#include <cstdint>
#include <vector>
#include <string>

namespace yichi {

enum class CellType : uint8_t {
    EMPTY = 0,
    X = 1,
    O = 2,
    BLOCK = 3
};

struct GameConfig {
    int board_size = 6;
    int initial_health = 2;
    int attack_power = 1;
    int heal_power = 1;
    bool diag_heal = true;
    bool diag_attack = true;
};

// Board: packed representation
//   types_:  N*N bytes (CellType)
//   health_: N*N int8_t (0..127)
struct Board {
    GameConfig config;
    std::vector<CellType> types;       // size = N*N
    std::vector<int8_t> health;        // size = N*N
    int current_player;                // 0 = X, 1 = O (matches CellType X=1, O=2 minus 1)
    int step = 0;
    bool board_changed = false;

    Board(const GameConfig& cfg);
    void reset();
    Board clone() const;
    bool operator==(const Board& other) const;

    // Coordinate helpers
    inline int idx(int r, int c) const { return r * config.board_size + c; }
    inline CellType at(int r, int c) const { return types[idx(r, c)]; }
    inline int8_t hp_at(int r, int c) const { return health[idx(r, c)]; }

    // Game operations
    bool is_empty(int r, int c) const;
    bool is_full() const;
    std::vector<std::pair<int,int>> legal_moves() const;
    bool is_terminal() const;
    int winner() const;  // 0=X, 1=O, -1=draw, 2=ongoing

    // Serialization (for caching)
    uint64_t hash() const;
    std::string to_string() const;
};

}  // namespace yichi
```

### 2.1 设计要点

- **紧凑布局**：`types` 用 `uint8_t` 而非 `int`，36 格只占 36 字节
- **hash()**：用于 transposition table 缓存 MCTS 评估结果
- **clone()**：MCTS 在每个子节点展开时需要状态副本

## 3. 规则实现 `rules.h`

```cpp
#pragma once
#include "board.h"

namespace yichi {

class Rules {
public:
    // Apply a move + chain reaction until stable
    static void apply_move(Board& board, int r, int c);

    // Individual rules (public for unit testing)
    static void heal_rule(Board& board, int r, int c);
    static void damage_rule(Board& board, int r, int c, int player);
    static void death_rule(Board& board, int r, int c, bool& changed);
    static void block_rule(Board& board);
    static void refresh_board(Board& board, bool& changed);

private:
    // Direct (up/down/left/right) and diagonal neighbor offsets
    static constexpr int DIRECT[4][2] = {{-1,0},{0,-1},{0,1},{1,0}};
    static constexpr int DIAGONAL[4][2] = {{-1,-1},{-1,1},{1,-1},{1,1}};

    static void count_neighbors(const Board& b, int r, int c,
                                int& n_same_direct, int& m_same_diag,
                                int& n_opp_direct, int& m_opp_diag,
                                int player);
};

}  // namespace yichi
```

### 3.1 apply_move 的关键实现

```cpp
void Rules::apply_move(Board& board, int r, int c) {
    // 1. Place stone
    int player = board.current_player;
    board.types[board.idx(r, c)] = (player == 0) ? CellType::X : CellType::O;
    board.health[board.idx(r, c)] = board.config.initial_health;

    // 2. Chain refresh until stable
    bool changed = false;
    do {
        changed = false;
        refresh_board(board, changed);
    } while (changed);

    // 3. Switch player
    board.current_player = 1 - player;
    board.step++;
}
```

### 3.2 refresh_board 的严格顺序

```cpp
void Rules::refresh_board(Board& board, bool& changed) {
    const int N = board.config.board_size;
    const auto& cfg = board.config;

    // Step A: block rule (only if asymmetric config)
    if (!cfg.diag_heal || !cfg.diag_attack || cfg.attack_power != cfg.heal_power) {
        block_rule(board);
    }

    // Step B: heal all
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            if (board.at(r, c) != CellType::EMPTY && board.at(r, c) != CellType::BLOCK)
                heal_rule(board, r, c);

    // Step C: damage + death for current player
    int cur = board.current_player;
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            if ((int)board.at(r, c) == cur + 1)
                damage_rule(board, r, c, cur);
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            if ((int)board.at(r, c) == cur + 1)
                death_rule(board, r, c, changed);

    // Step D: heal all again
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            if (board.at(r, c) != CellType::EMPTY && board.at(r, c) != CellType::BLOCK)
                heal_rule(board, r, c);

    // Step E: damage + death for opponent
    int opp = 1 - cur;
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            if ((int)board.at(r, c) == opp + 1)
                damage_rule(board, r, c, opp);
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            if ((int)board.at(r, c) == opp + 1)
                death_rule(board, r, c, changed);
}
```

完整实现见 `engine/src/rules.cpp`。

## 4. MCTS 实现 `mcts.h`

```cpp
#pragma once
#include "board.h"
#include "rules.h"
#include "neuralnet.h"
#include <memory>
#include <unordered_map>
#include <vector>
#include <atomic>

namespace yichi {

struct MCTSNode {
    Board state;
    MCTSNode* parent;
    std::pair<int,int> move;        // (-1,-1) for root
    float prior;                    // P(s, a)
    std::vector<std::unique_ptr<MCTSNode>> children;
    std::atomic<int> N{0};
    std::atomic<float> W{0.0f};
    float Q = 0.0f;
    bool is_expanded = false;
    std::atomic<int> virtual_loss{0};

    MCTSNode(Board s, MCTSNode* p, std::pair<int,int> m, float pr)
        : state(std::move(s)), parent(p), move(m), prior(pr) {}

    float ucb_score(float c_puct) const;
    MCTSNode* best_child(float c_puct) const;
};

class MCTS {
public:
    MCTS(YichiNet model, float c_puct = 1.5f,
         int n_simulations = 400,
         float dirichlet_alpha = 0.3f,
         float dirichlet_epsilon = 0.25f,
         int batch_size = 8);

    // Run MCTS from state, return root node (caller takes ownership)
    std::unique_ptr<MCTSNode> search(const Board& root_state);

    // Get action distribution after search
    std::vector<float> get_action_distribution(
        const MCTSNode& root, float temperature = 1.0f) const;

private:
    YichiNet model_;
    float c_puct_;
    int n_simulations_;
    float dirichlet_alpha_;
    float dirichlet_epsilon_;
    int batch_size_;

    // Batched evaluation queue
    std::vector<MCTSNode*> eval_queue_;

    void select_and_expand(MCTSNode* root);
    void backup(MCTSNode* node, float value);
    void evaluate_batch();
    void expand_node(MCTSNode* node,
                     const std::vector<float>& policy,
                     bool add_noise);
};

}  // namespace yichi
```

### 4.1 关键优化：批量推理

```cpp
void MCTS::select_and_expand(MCTSNode* root) {
    // 1. Select by PUCT until leaf
    MCTSNode* node = root;
    while (node->is_expanded && !node->state.is_terminal()) {
        node = node->best_child(c_puct_);
        // Apply virtual loss for multi-thread safety
        node->virtual_loss += 1;
    }

    // 2. If terminal, backup true reward
    if (node->state.is_terminal()) {
        int winner = node->state.winner();
        int cur_player = node->state.current_player;
        float value = (winner < 0) ? 0.0f :
                      (winner == cur_player) ? 1.0f : -1.0f;
        backup(node, value);
        return;
    }

    // 3. Add to batch queue
    eval_queue_.push_back(node);
    if ((int)eval_queue_.size() >= batch_size_) {
        evaluate_batch();
    }
}

void MCTS::evaluate_batch() {
    int n = eval_queue_.size();
    if (n == 0) return;

    // Build batched input tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor batch = torch::zeros({n, 11, board_size_, board_size_}, options);

    for (int i = 0; i < n; ++i) {
        board_to_tensor(eval_queue_[i]->state, batch[i]);
    }

    // Forward
    auto [policy_logits, value] = model_->forward(batch);
    auto policy = torch::softmax(policy_logits, /*dim=*/1);

    // Expand each node and backup
    auto policy_acc = policy.accessor<float, 2>();
    auto value_acc = value.accessor<float, 2>();

    for (int i = 0; i < n; ++i) {
        std::vector<float> p(board_size_ * board_size_ + 1);
        for (int j = 0; j < (int)p.size(); ++j) p[j] = policy_acc[i][j];

        bool add_noise = (eval_queue_[i] == root_);
        expand_node(eval_queue_[i], p, add_noise);
        backup(eval_queue_[i], value_acc[i][0]);

        // Clear virtual loss
        MCTSNode* n = eval_queue_[i];
        while (n != nullptr) {
            n->virtual_loss -= 1;
            n = n->parent;
        }
    }
    eval_queue_.clear();
}
```

## 5. 神经网络 `neuralnet.h`

```cpp
#pragma once
#include <torch/torch.h>
#include <utility>

namespace yichi {

class ResBlockImpl : public torch::nn::Module {
public:
    ResBlockImpl(int channels);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};
};
TORCH_MODULE(ResBlock);

class YichiNetImpl : public torch::nn::Module {
public:
    YichiNetImpl(int board_size, int in_channels = 11,
                 int channels = 64, int n_blocks = 6);

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

    // Load weights from a .pt file saved by Python training
    void load_from_python(const std::string& path);

private:
    int board_size_;
    int n_actions_;

    torch::nn::Sequential stem{nullptr};
    torch::nn::Sequential blocks{nullptr};
    torch::nn::Sequential policy_conv{nullptr};
    torch::nn::Linear policy_fc{nullptr};
    torch::nn::Sequential value_conv{nullptr};
    torch::nn::Linear value_fc1{nullptr};
    torch::nn::Linear value_fc2{nullptr};
};
TORCH_MODULE(YichiNet);

// Helper: convert Board to tensor (perspective-aligned)
torch::Tensor board_to_tensor(const Board& board);

}  // namespace yichi
```

### 5.1 跨语言权重加载

Python 训练时保存：

```python
torch.save({
    'state_dict': model.state_dict(),
    'board_size': 6,
    'in_channels': 11,
    'channels': 64,
    'n_blocks': 6,
}, 'model.pt')
```

C++ 加载：

```cpp
void YichiNetImpl::load_from_python(const std::string& path) {
    torch::load(*this, path);  // Simplified; actual loading uses torch::jit::load or state_dict
    this->eval();
}
```

> **注意**：跨 Python/C++ 加载需要 `torch.jit.script` 或严格的 `state_dict` 字典匹配。本框架推荐用 `torch.jit.script(model)` 保存为 TorchScript，C++ 端用 `torch::jit::load()` 加载，更兼容。

## 6. 自对弈驱动 `selfplay_main.cpp`

```cpp
#include "board.h"
#include "rules.h"
#include "mcts.h"
#include "neuralnet.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>

using namespace yichi;

struct SelfPlayConfig {
    std::string model_path;
    int n_games = 100;
    int n_threads = 8;
    int board_size = 6;
    int n_simulations = 400;
    std::string output_dir = "./selfplay_data";
};

void play_one_game(const SelfPlayConfig& cfg, YichiNet model, int thread_id) {
    GameConfig game_cfg;
    game_cfg.board_size = cfg.board_size;

    Board board(game_cfg);
    MCTS mcts(model, 1.5f, cfg.n_simulations);

    std::vector<std::pair<Board, std::vector<float>>> trajectory;

    while (!board.is_terminal()) {
        auto root = mcts.search(board);
        auto pi = mcts.get_action_distribution(*root, /*temperature=*/1.0f);

        trajectory.push_back({board.clone(), pi});

        // Sample action
        int action_idx = sample_from_distribution(pi, thread_id);
        int r = action_idx / cfg.board_size;
        int c = action_idx % cfg.board_size;
        Rules::apply_move(board, r, c);
    }

    // Backfill terminal value
    int winner = board.winner();
    save_game_to_file(trajectory, winner, cfg.output_dir, thread_id);
}

int main(int argc, char** argv) {
    SelfPlayConfig cfg = parse_args(argc, argv);

    // Load model once, share across threads
    YichiNet model(6);
    model->load_from_python(cfg.model_path);

    std::vector<std::thread> threads;
    for (int t = 0; t < cfg.n_threads; ++t) {
        threads.emplace_back([cfg, model, t]() {
            for (int g = 0; g < cfg.n_games / cfg.n_threads; ++g) {
                play_one_game(cfg, model, t);
            }
        });
    }
    for (auto& th : threads) th.join();

    return 0;
}
```

## 7. CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)
project(yichi_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# LibTorch (CPU version)
find_package(Torch REQUIRED)

add_executable(yichi_selfplay
    src/board.cpp
    src/rules.cpp
    src/mcts.cpp
    src/neuralnet.cpp
    src/selfplay_main.cpp
)

target_include_directories(yichi_selfplay PRIVATE include)
target_link_libraries(yichi_selfplay PRIVATE "${TORCH_LIBRARIES}")

# OpenMP for parallel self-play
find_package(OpenMP REQUIRED)
target_link_libraries(yichi_selfplay PRIVATE OpenMP::OpenMP_CXX)
```

## 8. 编译与运行

```bash
# 1. Install LibTorch (CPU version)
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.12.0%2Bcpu.zip
unzip libtorch-*.zip -d /opt/

# 2. Install CMake (if missing)
sudo apt install cmake

# 3. Build
cd engine/
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/opt/libtorch ..
make -j$(nproc)

# 4. Run self-play
./yichi_selfplay \
    --model /path/to/model.pt \
    --games 100 \
    --threads 8 \
    --board_size 6 \
    --output /path/to/selfplay_data/
```

## 9. 性能基准 (预期)

| 配置 | 自对弈速度 (局/秒) | 单步延迟 |
|------|-------------------|----------|
| Python (1 thread, 200 sim) | 0.03 | 30 s |
| Python (8 threads, 200 sim) | 0.2 | 5 s |
| C++ (1 thread, 400 sim) | 0.5 | 2 s |
| C++ (8 threads, 400 sim) | 3.0 | 0.3 s |
| C++ + GPU (8 threads, 800 sim) | 20+ | 0.05 s |

C++ 引擎相比 Python 约有 **10-15 倍** 性能提升，主要来自：

1. 紧凑的 Board 表示（避免 numpy overhead）
2. 批量推理（batch_size=8）
3. 多线程虚拟损失（无 GIL）
4. 内联化的规则计算

## 10. 与 Python 的对齐验证

C++ 引擎与 Python 实现必须严格一致。验证方法：

1. **单元测试**：1000 个随机局面，对每个局面执行 50 步随机落子，对比 Python 和 C++ 的最终状态，要求逐字节相同
2. **MCTS 一致性**：固定随机种子，让 Python 和 C++ MCTS 各跑 400 次模拟，对比访问次数分布，KL 散度 < 0.01
3. **网络一致性**：同一 `.pt` 模型，同一输入，Python 和 C++ 推理输出张量最大差异 < 1e-5

验证脚本：`scripts/verify_engine_alignment.py`（开发期使用，不在生产流程）。

## 11. 总结

C++ 引擎是本框架的"生产后端"，主要承担：

- **快速自对弈数据生成**：替代 Python 慢速自对弈
- **生产部署**：把训练好的模型集成到游戏网页（通过 WASM 或本地服务）
- **多线程扩展**：利用多核 CPU

Python pipeline 负责"研发迭代"——快速试错、调试、可视化。两者通过 `.pt` 权重文件解耦，可以独立演进。
