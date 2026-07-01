// engine/src/neuralnet.cpp
// YichiAlpha — Neural network implementation (LibTorch)

#include "neuralnet.h"

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/serialization/import.h>
#include <iostream>
#include <unordered_map>

namespace yichi {

// ---------------------------------------------------------------------------
// ResBlock
// ---------------------------------------------------------------------------
ResBlockImpl::ResBlockImpl(int channels) {
    conv1 = register_module("conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false)));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(channels));
    conv2 = register_module("conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false)));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(channels));
}

torch::Tensor ResBlockImpl::forward(torch::Tensor x) {
    auto identity = x;
    auto out = torch::relu(bn1(conv1(x)));
    out = bn2(conv2(out));
    out = torch::relu(out + identity);
    return out;
}

// ---------------------------------------------------------------------------
// YichiNet
// ---------------------------------------------------------------------------
YichiNetImpl::YichiNetImpl(int board_size, int in_channels, int channels, int n_blocks)
    : board_size_(board_size),
      in_channels_(in_channels),
      channels_(channels),
      n_blocks_(n_blocks),
      n_actions_(board_size * board_size + 1) {

    // Stem
    stem = register_module("stem", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, channels, 3).padding(1).bias(false)),
        torch::nn::BatchNorm2d(channels),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
    ));

    // Residual blocks
    torch::nn::Sequential blocks_seq;
    for (int i = 0; i < n_blocks; ++i) {
        blocks_seq->push_back(ResBlock(channels));
    }
    blocks = register_module("blocks", blocks_seq);

    // Policy head
    policy_conv = register_module("policy_conv", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, 2, 1).bias(false)),
        torch::nn::BatchNorm2d(2),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
    ));
    policy_fc = register_module("policy_fc",
        torch::nn::Linear(2 * board_size * board_size, n_actions_));

    // Value head
    value_conv = register_module("value_conv", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, 1, 1).bias(false)),
        torch::nn::BatchNorm2d(1),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
    ));
    value_fc1 = register_module("value_fc1",
        torch::nn::Linear(board_size * board_size, 32));
    value_fc2 = register_module("value_fc2",
        torch::nn::Linear(32, 1));

    init_weights();
}

void YichiNetImpl::init_weights() {
    // Use modules(/*include_self=*/false) to avoid shared_from_this issue
    // when called from the constructor (before the object is in a shared_ptr).
    for (auto& m : modules(/*include_self=*/false)) {
        if (auto* conv = m->as<torch::nn::Conv2d>()) {
            torch::nn::init::kaiming_normal_(conv->weight);
        } else if (auto* lin = m->as<torch::nn::Linear>()) {
            torch::nn::init::xavier_uniform_(lin->weight);
            if (lin->bias.defined()) {
                torch::nn::init::zeros_(lin->bias);
            }
        } else if (auto* bn = m->as<torch::nn::BatchNorm2d>()) {
            torch::nn::init::ones_(bn->weight);
            torch::nn::init::zeros_(bn->bias);
        }
    }
    // Shrink final layers (must use NoGradGuard since params require grad by default)
    {
        torch::NoGradGuard no_grad;
        policy_fc->weight.mul_(0.01);
        value_fc2->weight.mul_(0.01);
    }
}

std::pair<torch::Tensor, torch::Tensor> YichiNetImpl::forward(torch::Tensor x) {
    auto h = stem->forward(x);
    h = blocks->forward(h);

    // Policy
    auto p = policy_conv->forward(h);
    p = p.view({p.size(0), -1});
    p = policy_fc->forward(p);

    // Value
    auto v = value_conv->forward(h);
    v = v.view({v.size(0), -1});
    v = torch::relu(value_fc1->forward(v));
    v = torch::tanh(value_fc2->forward(v));

    return {p, v};
}

void YichiNetImpl::load_from_python(const std::string& path) {
    // Load a TorchScript-scripted model saved via torch.jit.script(model).save(path, extra_files).
    // We extract parameters from the TorchScript module and copy them into our module.
    std::shared_ptr<torch::jit::Module> jit_module;
    try {
        // Read extra_files for config
        torch::jit::ExtraFilesMap extra_files = {
            {"board_size.txt", ""}, {"in_channels.txt", ""},
            {"channels.txt", ""}, {"n_blocks.txt", ""},
        };
        jit_module = std::make_shared<torch::jit::Module>(
            torch::jit::load(path, /*device=*/torch::nullopt, extra_files)
        );
        jit_module->eval();
        std::cout << "TorchScript model loaded from " << path << std::endl;

        // Print embedded config
        for (const auto& kv : extra_files) {
            if (!kv.second.empty()) {
                std::cout << "  " << kv.first << " = " << kv.second << std::endl;
            }
        }

        // Verify dimensions match
        if (!extra_files["channels.txt"].empty()) {
            int file_channels = std::stoi(extra_files["channels.txt"]);
            if (file_channels != channels_) {
                std::cerr << "WARNING: Model file has channels=" << file_channels
                          << " but C++ module was built with channels=" << channels_
                          << ". Weights will not be loaded correctly." << std::endl;
            }
        }
        if (!extra_files["n_blocks.txt"].empty()) {
            int file_blocks = std::stoi(extra_files["n_blocks.txt"]);
            if (file_blocks != n_blocks_) {
                std::cerr << "WARNING: Model file has n_blocks=" << file_blocks
                          << " but C++ module was built with n_blocks=" << n_blocks_
                          << ". Weights will not be loaded correctly." << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: failed to load TorchScript model from " << path
                  << ": " << e.what() << std::endl;
        std::cerr << "Continuing with random-initialized weights." << std::endl;
        return;
    }

    // Build a map from name -> Tensor for the JIT module (params + buffers)
    std::unordered_map<std::string, torch::Tensor> jit_map;
    for (const auto& p : jit_module->named_parameters()) {
        jit_map[p.name] = p.value;
    }
    for (const auto& b : jit_module->named_buffers()) {
        jit_map[b.name] = b.value;
    }

    int loaded = 0, skipped = 0;
    torch::NoGradGuard no_grad;
    for (auto& p : named_parameters()) {
        auto it = jit_map.find(p.key());
        if (it != jit_map.end() && it->second.sizes() == p.value().sizes()) {
            p.value().copy_(it->second);
            ++loaded;
        } else {
            ++skipped;
        }
    }
    for (auto& b : named_buffers()) {
        auto it = jit_map.find(b.key());
        if (it != jit_map.end() && it->second.sizes() == b.value().sizes()) {
            b.value().copy_(it->second);
            ++loaded;
        } else {
            ++skipped;
        }
    }
    std::cout << "Loaded " << loaded << " params/buffers, skipped " << skipped << std::endl;
    this->eval();
}

}  // namespace yichi
