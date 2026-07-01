// engine/include/neuralnet.h
// YichiAlpha — Policy-Value Network (LibTorch C++)
//
// Mirrors python/model.py::YichiNet exactly. Weights saved by Python
// training can be loaded here via torch::load (state_dict) or TorchScript.

#pragma once

#include <torch/torch.h>
#include <utility>
#include <string>

namespace yichi {

// Residual block: Conv-BN-ReLU-Conv-BN-Add-ReLU
class ResBlockImpl : public torch::nn::Module {
public:
    ResBlockImpl(int channels);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d     conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Conv2d     conv2{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};
};
TORCH_MODULE(ResBlock);

// Policy-Value network
class YichiNetImpl : public torch::nn::Module {
public:
    YichiNetImpl(int board_size, int in_channels = 11,
                 int channels = 64, int n_blocks = 6);

    // Returns (policy_logits, value)
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

    // Load weights from a .pt file saved by Python training.
    // The Python side saves a dict with 'state_dict' and config keys;
    // we expect a TorchScript-scripted model OR a state_dict file.
    void load_from_python(const std::string& path);

    int board_size() const { return board_size_; }
    int n_actions() const { return n_actions_; }

private:
    int board_size_;
    int in_channels_;
    int channels_;
    int n_blocks_;
    int n_actions_;

    torch::nn::Sequential stem{nullptr};
    torch::nn::Sequential blocks{nullptr};
    torch::nn::Sequential policy_conv{nullptr};
    torch::nn::Linear     policy_fc{nullptr};
    torch::nn::Sequential value_conv{nullptr};
    torch::nn::Linear     value_fc1{nullptr};
    torch::nn::Linear     value_fc2{nullptr};

    void init_weights();
};
TORCH_MODULE(YichiNet);

}  // namespace yichi
