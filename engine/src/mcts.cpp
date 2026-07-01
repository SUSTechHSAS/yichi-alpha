// engine/src/mcts.cpp
// YichiAlpha — MCTS implementation (C++)

#include "mcts.h"
#include "rules.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <stdexcept>

namespace yichi {

// ---------------------------------------------------------------------------
// MCTSNode
// ---------------------------------------------------------------------------
MCTSNode::MCTSNode(Board s, MCTSNode* p, std::pair<int,int> m, float pr)
    : state(std::move(s)), parent(p), move(m), prior(pr) {}

float MCTSNode::ucb_score(float c_puct) const {
    if (parent == nullptr) return 0.0f;
    float u = c_puct * prior *
              (std::sqrt(static_cast<float>(parent->N.load())) /
               (1.0f + static_cast<float>(N.load())));
    // Q is stored from "this node's player's perspective".
    // Parent's player is the opponent, so parent's value = -Q.
    return -Q + u;
}

MCTSNode* MCTSNode::best_child(float c_puct) const {
    MCTSNode* best = nullptr;
    float best_score = -1e9f;
    for (auto& c : children) {
        float s = c->ucb_score(c_puct);
        if (s > best_score) {
            best_score = s;
            best = c.get();
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// MCTS
// ---------------------------------------------------------------------------
MCTS::MCTS(YichiNet model, float c_puct, int n_simulations,
           float dirichlet_alpha, float dirichlet_epsilon, int batch_size,
           torch::Device device)
    : model_(std::move(model)),
      c_puct_(c_puct),
      n_simulations_(n_simulations),
      dirichlet_alpha_(dirichlet_alpha),
      dirichlet_epsilon_(dirichlet_epsilon),
      batch_size_(batch_size),
      board_size_(0),
      device_(device) {
    // board_size_ will be set per search from root_state
    // Move model to the target device (CPU or CUDA)
    model_->to(device_);
}

torch::Tensor board_to_tensor(const Board& board, torch::Device device) {
    const int N = board.board_size();
    const int p = board.current_player();
    const int opp = (p == 1) ? 2 : 1;

    // Build on CPU first (faster for small tensor initialization)
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor planes = torch::zeros({1, 11, N, N}, options);
    auto accessor = planes.accessor<float, 4>();
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            int t = static_cast<int>(board.at(r, c));
            int h = board.hp_at(r, c);
            if (t == p) {
                accessor[0][0][r][c] = 1.0f;
                if (h == 1)       accessor[0][3][r][c] = 1.0f;
                else if (h == 2)  accessor[0][4][r][c] = 1.0f;
                else if (h >= 3)  accessor[0][5][r][c] = 1.0f;
            } else if (t == opp) {
                accessor[0][1][r][c] = 1.0f;
                if (h == 1)       accessor[0][6][r][c] = 1.0f;
                else if (h == 2)  accessor[0][7][r][c] = 1.0f;
                else if (h >= 3)  accessor[0][8][r][c] = 1.0f;
            } else if (t == 3) {  // BLOCK
                accessor[0][2][r][c] = 1.0f;
            }
        }
    }
    // Legal moves mask
    auto legal = board.legal_moves();
    for (auto& mv : legal) {
        accessor[0][9][mv.first][mv.second] = 1.0f;
    }
    // Side to move (constant 1 after perspective alignment)
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            accessor[0][10][r][c] = 1.0f;

    // Move to target device if not CPU
    if (device.is_cuda()) {
        planes = planes.to(device);
    }
    return planes;
}

void MCTS::expand_node(MCTSNode* node, const std::vector<float>& policy, bool add_noise) {
    auto legal = node->state.legal_moves();
    const int N = node->state.board_size();

    std::vector<float> priors;
    priors.reserve(legal.size());
    for (auto& mv : legal) {
        int idx = mv.first * N + mv.second;
        priors.push_back(policy[idx]);
    }

    if (add_noise && !legal.empty()) {
        std::gamma_distribution<float> gamma(dirichlet_alpha_, 1.0f);
        std::mt19937 rng(std::random_device{}());
        std::vector<float> noise(legal.size());
        float noise_sum = 0.0f;
        for (size_t i = 0; i < legal.size(); ++i) {
            noise[i] = gamma(rng);
            noise_sum += noise[i];
        }
        if (noise_sum > 0) {
            for (size_t i = 0; i < legal.size(); ++i) {
                priors[i] = (1.0f - dirichlet_epsilon_) * priors[i] +
                            dirichlet_epsilon_ * (noise[i] / noise_sum);
            }
        }
    }

    // Renormalize
    float sum = 0.0f;
    for (float p : priors) sum += p;
    if (sum > 0) {
        for (float& p : priors) p /= sum;
    } else {
        std::fill(priors.begin(), priors.end(), 1.0f / priors.size());
    }

    for (size_t i = 0; i < legal.size(); ++i) {
        Board child = node->state.clone();
        Rules::apply_move(child, legal[i].first, legal[i].second);
        node->children.push_back(std::make_unique<MCTSNode>(
            std::move(child), node, legal[i], priors[i]));
    }
    node->is_expanded = true;
}

void MCTS::backup(MCTSNode* node, float value) {
    while (node != nullptr) {
        node->N.fetch_add(1);
        float old_w = node->W.load();
        while (!node->W.compare_exchange_weak(old_w, old_w + value)) {}
        int n = node->N.load();
        node->Q = node->W.load() / static_cast<float>(n);
        value = -value;  // flip perspective
        node = node->parent;
    }
}

void MCTS::evaluate_batch(bool root_add_noise) {
    int n = static_cast<int>(eval_queue_.size());
    if (n == 0) return;

    // Build batched input (on CPU, then move to device_ once)
    std::vector<torch::Tensor> inputs;
    inputs.reserve(n);
    for (int i = 0; i < n; ++i) {
        inputs.push_back(board_to_tensor(eval_queue_[i]->state, torch::kCPU));
    }
    torch::Tensor batch = torch::cat(inputs, /*dim=*/0);
    batch = batch.to(device_);

    // Forward
    auto [logits, value] = model_->forward(batch);
    auto policy = torch::softmax(logits, /*dim=*/1);

    // Move outputs to CPU for accessor (works regardless of device)
    policy = policy.to(torch::kCPU);
    value = value.to(torch::kCPU);

    auto policy_acc = policy.accessor<float, 2>();
    auto value_acc = value.accessor<float, 2>();
    int N = eval_queue_[0]->state.board_size();

    for (int i = 0; i < n; ++i) {
        std::vector<float> p(N * N + 1);
        for (int j = 0; j < N * N + 1; ++j) p[j] = policy_acc[i][j];

        bool add_noise = (eval_queue_[i]->parent == nullptr) && root_add_noise;
        expand_node(eval_queue_[i], p, add_noise);
        backup(eval_queue_[i], value_acc[i][0]);
    }
    eval_queue_.clear();
}

void MCTS::select_and_expand(MCTSNode* root, bool root_add_noise) {
    MCTSNode* node = root;
    while (node->is_expanded && !node->state.is_terminal()) {
        node = node->best_child(c_puct_);
    }

    if (node->state.is_terminal()) {
        float v = node->state.terminal_reward_for_current_player();
        backup(node, v);
        return;
    }

    eval_queue_.push_back(node);
    if (static_cast<int>(eval_queue_.size()) >= batch_size_) {
        evaluate_batch(root_add_noise);
    }
}

std::unique_ptr<MCTSNode> MCTS::search(const Board& root_state, bool add_noise) {
    board_size_ = root_state.board_size();
    auto root = std::make_unique<MCTSNode>(root_state.clone(), nullptr,
                                            std::make_pair(-1, -1), 0.0f);

    // Evaluate root
    auto input = board_to_tensor(root->state, device_);
    auto [logits, value] = model_->forward(input);
    auto policy = torch::softmax(logits, /*dim=*/1);
    // Move policy back to CPU for accessor (always works regardless of device)
    policy = policy.to(torch::kCPU);
    value = value.to(torch::kCPU);
    auto policy_acc = policy.accessor<float, 2>();
    int N = board_size_;
    std::vector<float> p(N * N + 1);
    for (int j = 0; j < N * N + 1; ++j) p[j] = policy_acc[0][j];
    expand_node(root.get(), p, add_noise);

    if (root->state.is_terminal()) return root;

    for (int i = 0; i < n_simulations_; ++i) {
        select_and_expand(root.get(), add_noise);
    }
    // Flush any remaining
    evaluate_batch(add_noise);
    return root;
}

std::vector<float> MCTS::get_action_distribution(
    const MCTSNode& root, float temperature) const {

    const int N = root.state.board_size();
    std::vector<float> visits(N * N + 1, 0.0f);

    for (const auto& c : root.children) {
        int idx = c->move.first * N + c->move.second;
        visits[idx] = static_cast<float>(c->N.load());
    }

    if (temperature == 0.0f) {
        int max_idx = 0;
        float max_v = -1.0f;
        for (int i = 0; i < N * N + 1; ++i) {
            if (visits[i] > max_v) {
                max_v = visits[i];
                max_idx = i;
            }
        }
        std::vector<float> result(N * N + 1, 0.0f);
        result[max_idx] = 1.0f;
        return result;
    }

    float sum = 0.0f;
    for (float& v : visits) {
        v = std::pow(v, 1.0f / temperature);
        sum += v;
    }
    if (sum > 0) {
        for (float& v : visits) v /= sum;
    } else {
        std::fill(visits.begin(), visits.end(), 1.0f / visits.size());
    }
    return visits;
}

std::pair<int,int> MCTS::select_action(
    const MCTSNode& root, float temperature, unsigned long seed) const {

    auto visits = get_action_distribution(root, temperature);
    const int N = root.state.board_size();

    std::mt19937 rng(seed ? seed : std::random_device{}());
    std::discrete_distribution<int> dist(visits.begin(), visits.end());
    int idx = dist(rng);

    if (idx >= N * N) {
        // Pass fallback
        auto legal = root.state.legal_moves();
        return legal.empty() ? std::make_pair(-1, -1) : legal[0];
    }
    return {idx / N, idx % N};
}

}  // namespace yichi
