// engine/src/selfplay_main.cpp
// YichiAlpha — Self-play driver (C++)
//
// Generates self-play games using the C++ engine + LibTorch neural network.
// Output: one file per game, containing serialized (state, policy, value) samples.
//
// Usage:
//   ./yichi_selfplay --model path/to/model.pt --games 100 --threads 8 \
//                    --board_size 6 --output ./selfplay_data/

#include "board.h"
#include "rules.h"
#include "mcts.h"
#include "neuralnet.h"

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/serialization/import.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace yichi;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
struct SelfPlayConfig {
    std::string model_path;
    int n_games = 100;
    int n_threads = 8;
    int board_size = 6;
    int n_simulations = 400;
    float c_puct = 1.5f;
    float dirichlet_alpha = 0.3f;
    float dirichlet_epsilon = 0.25f;
    int batch_size = 8;
    std::string output_dir = "./selfplay_data";
};

SelfPlayConfig parse_args(int argc, char** argv) {
    SelfPlayConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << a << std::endl;
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--model") cfg.model_path = next();
        else if (a == "--games") cfg.n_games = atoi(next().c_str());
        else if (a == "--threads") cfg.n_threads = atoi(next().c_str());
        else if (a == "--board_size") cfg.board_size = atoi(next().c_str());
        else if (a == "--n_simulations") cfg.n_simulations = atoi(next().c_str());
        else if (a == "--c_puct") cfg.c_puct = atof(next().c_str());
        else if (a == "--output") cfg.output_dir = next();
        else if (a == "--batch_size") cfg.batch_size = atoi(next().c_str());
        else {
            std::cerr << "Unknown arg: " << a << std::endl;
            std::exit(1);
        }
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Game serializer
// ---------------------------------------------------------------------------
struct GameRecord {
    // Each entry: (state_bytes, policy_vector, value_scalar, player)
    std::vector<std::string> state_snapshots;
    std::vector<std::vector<float>> policies;
    std::vector<float> values;
    std::vector<int> players;
    int winner;
};

void save_game(const GameRecord& record, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open " << path << std::endl;
        return;
    }
    int n_samples = static_cast<int>(record.state_snapshots.size());
    out.write(reinterpret_cast<const char*>(&n_samples), sizeof(int));
    out.write(reinterpret_cast<const char*>(&record.winner), sizeof(int));

    for (int i = 0; i < n_samples; ++i) {
        // State bytes: N*N types + N*N health = 2*N*N bytes
        const auto& sb = record.state_snapshots[i];
        int sb_size = static_cast<int>(sb.size());
        out.write(reinterpret_cast<const char*>(&sb_size), sizeof(int));
        out.write(sb.data(), sb_size);

        // Policy vector (length = N*N+1)
        int p_size = static_cast<int>(record.policies[i].size());
        out.write(reinterpret_cast<const char*>(&p_size), sizeof(int));
        out.write(reinterpret_cast<const char*>(record.policies[i].data()),
                  p_size * sizeof(float));

        // Value
        out.write(reinterpret_cast<const char*>(&record.values[i]), sizeof(float));
        // Player
        out.write(reinterpret_cast<const char*>(&record.players[i]), sizeof(int));
    }
    out.close();
}

std::string board_to_bytes(const Board& b) {
    // Serialize: N, current_player, step, then N*N types (1 byte each) + N*N health (1 byte each)
    std::ostringstream oss;
    int N = b.board_size();
    oss.put(static_cast<char>(N));
    oss.put(static_cast<char>(b.current_player()));
    oss.put(static_cast<char>(b.step()));
    for (int i = 0; i < N * N; ++i) {
        oss.put(static_cast<char>(static_cast<int>(b.types()[i])));
    }
    for (int i = 0; i < N * N; ++i) {
        oss.put(static_cast<char>(b.health()[i]));
    }
    return oss.str();
}

// ---------------------------------------------------------------------------
// Self-play one game
// ---------------------------------------------------------------------------
GameRecord play_one_game(const SelfPlayConfig& cfg, YichiNet& model, int thread_id, int game_id) {
    GameConfig game_cfg;
    game_cfg.board_size = cfg.board_size;

    Board board(game_cfg);
    MCTS mcts(model, cfg.c_puct, cfg.n_simulations,
              cfg.dirichlet_alpha, cfg.dirichlet_epsilon, cfg.batch_size);

    GameRecord record;
    int max_steps = cfg.board_size * cfg.board_size + 5;

    for (int step = 0; step < max_steps && !board.is_terminal(); ++step) {
        auto root = mcts.search(board, /*add_noise=*/true);
        auto pi = mcts.get_action_distribution(*root, /*temperature=*/1.0f);

        // Snapshot state BEFORE applying move
        record.state_snapshots.push_back(board_to_bytes(board));
        record.policies.push_back(pi);
        record.players.push_back(board.current_player());

        // Sample action (use thread-local RNG via static thread_local)
        static thread_local std::mt19937 rng(std::random_device{}());
        std::discrete_distribution<int> dist(pi.begin(), pi.end());
        int idx = dist(rng);

        int N = board.board_size();
        if (idx >= N * N) {
            auto legal = board.legal_moves();
            if (legal.empty()) break;
            board.apply_move(legal[0].first, legal[0].second);
        } else {
            board.apply_move(idx / N, idx % N);
        }
    }

    record.winner = board.winner();

    // Backfill values
    record.values.resize(record.players.size());
    for (size_t i = 0; i < record.players.size(); ++i) {
        if (record.winner == -1) {
            record.values[i] = 0.0f;
        } else if (record.winner == record.players[i]) {
            record.values[i] = 1.0f;
        } else {
            record.values[i] = -1.0f;
        }
    }
    return record;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    auto cfg = parse_args(argc, argv);

    if (cfg.model_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --model <path> [--games N] [--threads N] ..." << std::endl;
        return 1;
    }

    std::cout << "YichiAlpha self-play driver" << std::endl;
    std::cout << "  Model: " << cfg.model_path << std::endl;
    std::cout << "  Games: " << cfg.n_games << std::endl;
    std::cout << "  Threads: " << cfg.n_threads << std::endl;
    std::cout << "  Board size: " << cfg.board_size << std::endl;
    std::cout << "  MCTS simulations: " << cfg.n_simulations << std::endl;
    std::cout << "  Output dir: " << cfg.output_dir << std::endl;

    // Create output dir
    std::string mkdir_cmd = "mkdir -p " + cfg.output_dir;
    std::system(mkdir_cmd.c_str());

    // Load model once, share across threads.
    // First peek at the TorchScript file's extra_files metadata to get config,
    // then construct the model with matching dimensions.
    int file_board_size = cfg.board_size;
    int file_channels = 64;
    int file_n_blocks = 6;
    int file_in_channels = 11;
    try {
        torch::jit::ExtraFilesMap extra_files = {
            {"board_size.txt", ""}, {"in_channels.txt", ""},
            {"channels.txt", ""}, {"n_blocks.txt", ""},
        };
        auto peek = torch::jit::load(cfg.model_path, torch::nullopt, extra_files);
        (void)peek;  // we just want the metadata
        if (!extra_files["board_size.txt"].empty())   file_board_size   = std::stoi(extra_files["board_size.txt"]);
        if (!extra_files["in_channels.txt"].empty())  file_in_channels  = std::stoi(extra_files["in_channels.txt"]);
        if (!extra_files["channels.txt"].empty())     file_channels     = std::stoi(extra_files["channels.txt"]);
        if (!extra_files["n_blocks.txt"].empty())     file_n_blocks     = std::stoi(extra_files["n_blocks.txt"]);
        std::cout << "Model config from file: board_size=" << file_board_size
                  << ", in_channels=" << file_in_channels
                  << ", channels=" << file_channels
                  << ", n_blocks=" << file_n_blocks << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: could not peek model config, using CLI defaults: " << e.what() << std::endl;
    }

    YichiNet model = std::make_shared<YichiNetImpl>(
        file_board_size, file_in_channels, file_channels, file_n_blocks
    );
    try {
        model->load_from_python(cfg.model_path);
        std::cout << "Model loaded successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        std::cerr << "Continuing with random-initialized model for testing." << std::endl;
    }

    std::atomic<int> games_done{0};
    std::atomic<int> total_samples{0};
    auto t_start = std::chrono::steady_clock::now();

    auto worker = [&](int thread_id) {
        int games_per_thread = cfg.n_games / cfg.n_threads;
        if (thread_id < cfg.n_games % cfg.n_threads) games_per_thread++;

        for (int g = 0; g < games_per_thread; ++g) {
            int game_id = thread_id * 1000 + g;
            auto record = play_one_game(cfg, model, thread_id, game_id);

            std::string path = cfg.output_dir + "/game_" +
                               std::to_string(game_id) + ".bin";
            save_game(record, path);

            int done = games_done.fetch_add(1) + 1;
            total_samples.fetch_add(static_cast<int>(record.state_snapshots.size()));

            if (done % 5 == 0 || done == cfg.n_games) {
                auto t_now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(t_now - t_start).count();
                std::cout << "[" << done << "/" << cfg.n_games << "] "
                          << total_samples.load() << " samples, "
                          << elapsed << "s elapsed, "
                          << (done / elapsed) << " games/s" << std::endl;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < cfg.n_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& th : threads) th.join();

    auto t_end = std::chrono::steady_clock::now();
    double total = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "\nDone. " << cfg.n_games << " games, "
              << total_samples.load() << " samples, "
              << total << "s (" << (cfg.n_games / total) << " games/s)" << std::endl;

    return 0;
}
