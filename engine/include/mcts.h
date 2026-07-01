// engine/include/mcts.h
// YichiAlpha — MCTS with neural network guidance
//
// AlphaZero-style MCTS adapted for 异吃棋 chain reactions.
// See docs/04_mcts_adaptation.md for design rationale.

#pragma once

#include "board.h"
#include "rules.h"
#include "neuralnet.h"

#include <atomic>
#include <memory>
#include <unordered_map>
#include <vector>
#include <utility>

namespace yichi {

struct MCTSNode {
    Board state;
    MCTSNode* parent;
    std::pair<int,int> move;        // (-1, -1) for root
    float prior;
    std::vector<std::unique_ptr<MCTSNode>> children;
    std::atomic<int> N{0};
    std::atomic<float> W{0.0f};
    float Q = 0.0f;
    bool is_expanded = false;
    std::atomic<int> virtual_loss{0};

    MCTSNode(Board s, MCTSNode* p, std::pair<int,int> m, float pr);

    float ucb_score(float c_puct) const;
    MCTSNode* best_child(float c_puct) const;
};

class MCTS {
public:
    MCTS(YichiNet model,
         float c_puct = 1.5f,
         int n_simulations = 200,
         float dirichlet_alpha = 0.3f,
         float dirichlet_epsilon = 0.25f,
         int batch_size = 8,
         torch::Device device = torch::kCPU);

    // Run MCTS from root_state, return root node.
    std::unique_ptr<MCTSNode> search(const Board& root_state, bool add_noise = true);

    // Get action distribution (length N*N+1).
    std::vector<float> get_action_distribution(
        const MCTSNode& root, float temperature = 1.0f) const;

    // Select an action by sampling from visit distribution.
    std::pair<int,int> select_action(
        const MCTSNode& root, float temperature = 1.0f,
        unsigned long seed = 0) const;

private:
    YichiNet model_;
    float c_puct_;
    int n_simulations_;
    float dirichlet_alpha_;
    float dirichlet_epsilon_;
    int batch_size_;
    int board_size_;
    torch::Device device_;   // kCPU or kCUDA

    // Batched evaluation queue
    std::vector<MCTSNode*> eval_queue_;

    void select_and_expand(MCTSNode* root, bool root_add_noise);
    void backup(MCTSNode* node, float value);
    void evaluate_batch(bool root_add_noise);
    void expand_node(MCTSNode* node,
                     const std::vector<float>& policy,
                     bool add_noise);
};

// Helper: convert Board to network input tensor (perspective-aligned).
// Output shape: (1, 11, N, N)
// If device is kCUDA, the tensor is moved to GPU.
torch::Tensor board_to_tensor(const Board& board, torch::Device device = torch::kCPU);

}  // namespace yichi
