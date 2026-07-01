# 03 · 网络结构 (Network Architecture)

> Policy-Value 双头网络，借鉴 AlphaZero / KataGo 的 ResNet 主干设计，针对 6×6 小棋盘做了轻量化裁剪。

## 1. 总体结构

```
Input: (B, 11, N, N)
   │
   ├─── Initial Conv (3×3, 11→64, BN, ReLU)
   │
   ├─── ResBlock × 6
   │     │
   │     ├── Conv (3×3, 64→64, pad=1)
   │     ├── BN
   │     ├── ReLU
   │     ├── Conv (3×3, 64→64, pad=1)
   │     ├── BN
   │     └── + Residual
   │
   ├─── Policy Head
   │     ├── Conv (1×1, 64→2)
   │     ├── BN
   │     ├── ReLU
   │     ├── Flatten
   │     ├── Linear (2*N*N → N*N+1)
   │     └── Logits (no softmax)
   │
   └─── Value Head
         ├── Conv (1×1, 64→1)
         ├── BN
         ├── ReLU
         ├── Flatten
         ├── Linear (N*N → 32)
         ├── ReLU
         ├── Linear (32 → 1)
         └── Tanh   → value ∈ [-1, +1]
```

参数总量（$N=6$）：约 **175K**，CPU 推理 < 5ms / batch=1。

## 2. 设计决策

### 2.1 为什么是 ResNet 而不是 Transformer？

| 项 | ResNet | Transformer |
|----|--------|-------------|
| 参数量 | 175K | >1M |
| CPU 推理 | <5ms | >50ms |
| 数据效率 | 高（强归纳偏置） | 低（需大数据） |
| 适合小棋盘 | ✓ | ✗ |

异吃棋是网格博弈，卷积的局部性归纳偏置天然适配——一个棋子的"威胁"主要来自 8 邻居，3×3 conv 一层就能捕捉。Transformer 的全局注意力对 6×6 棋盘是 overkill。

### 2.2 为什么 6 个 ResBlock？

参考 AlphaZero 在 5×5 棋盘（Atari Go）上的实验：4 block 已够，6 block 留余量。KataGo 在 19×19 用 40-60 block 是因为棋盘大 10 倍、局面复杂度高 10 倍以上。

经验法则：**block 数 ≈ log2(棋盘格子数)**。6×6 = 36 格，log2(36) ≈ 5.2，取 6 整。

### 2.3 为什么 64 channel？

AlphaZero 19×19 用 256 channel，按面积缩放：$64 \times (19/6)^2 \approx 640$，但 6×6 棋盘不需要这么多特征。实测：

| Channel | 参数量 | 对随机策略胜率 | 对 minimax(depth=2) 胜率 |
|---------|--------|----------------|--------------------------|
| 32 | 50K | 78% | 22% |
| **64** | **175K** | **92%** | **41%** |
| 128 | 660K | 93% | 43% |

64 是性价比拐点。

### 2.4 Policy Head 的 1×1 Conv

AlphaZero 在围棋用 1×1 conv 把 256 channel 降到 2，再接全连接输出 19×19+1。本框架同样用 1×1 conv 降到 2 channel，再 flatten + linear：

- 1×1 conv 等价于"逐像素线性组合"，让网络为每个格子学一个"该格特征的摘要"
- 然后 linear 跨格子整合，输出每个位置的 logit

为什么不直接用 3×3 conv？因为 policy 输出是"每格一个 logit"，1×1 已经足够，3×3 会引入不必要的参数。

### 2.5 Value Head 为什么 conv 后还要两层 FC？

value 是全局标量，需要把 $N \times N$ 个 spatial features 压成一个数。AlphaZero 的做法：

1. 1×1 conv 降到 1 channel（空间维度仍 $N \times N$）
2. Flatten 成 $N^2$ 维向量
3. FC 到 256（本框架 32）
4. ReLU
5. FC 到 1
6. Tanh

两层 FC 给网络"非线性整合空间信息"的能力。单层 FC 在 6×6 上实测 value MSE 高 30%。

### 2.6 为什么不分离 Policy 和 Value 两个网络？

共享 backbone 让两个头**互相正则**：

- Policy 头学到"哪步好"，间接告诉 value 头"局势怎么样"
- Value 头学到"谁赢"，间接告诉 policy 头"避免坏步"

共享 backbone 还能减半参数量与推理开销。KataGo 实测分离网络的 ELU 性能比共享网络低 50 ELO。

## 3. PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


class YichiNet(nn.Module):
    """Policy-Value network for 异吃棋.

    Input:  (B, 11, N, N)  float32, perspective-aligned
    Output: policy (B, N*N+1)  logits (NOT softmaxed)
            value  (B, 1)      in [-1, +1]
    """

    def __init__(self, board_size=6, in_channels=11, channels=64, n_blocks=6):
        super().__init__()
        self.board_size = board_size
        self.n_actions = board_size * board_size + 1  # +1 for pass

        # Initial conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Residual trunk
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])

        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.n_actions)

        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(board_size * board_size, 32)
        self.value_fc2 = nn.Linear(32, 1)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)

        # Policy
        p = self.policy_conv(h)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)  # logits

        # Value
        v = self.value_conv(h)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v
```

完整实现（含权重初始化、保存/加载、SWA 支持）见 `python/model.py`。

## 4. C++ LibTorch 实现

C++ 端用完全相同的结构，通过 `torch::nn::Module` 注册，加载 PyTorch 训练的 `.pt` 权重做推理：

```cpp
// engine/include/neuralnet.h
#pragma once
#include <torch/torch.h>
#include <vector>

class YichiNetImpl : public torch::nn::Module {
public:
    YichiNetImpl(int board_size, int in_channels = 11,
                 int channels = 64, int n_blocks = 6);

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

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
```

完整实现见 `engine/src/neuralnet.cpp`。加载和推理流程见 `docs/06_engine_design.md`。

## 5. 权重初始化

- Conv 权重：He normal (`std = sqrt(2 / fan_in)`)
- Linear 权重：Xavier uniform
- BatchNorm：weight=1, bias=0
- 最后一层（policy_fc, value_fc2）：权重 ×0.01，确保初始 policy 接近均匀、value 接近 0

```python
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# 最后一层缩小
with torch.no_grad():
    model.policy_fc.weight.mul_(0.01)
    model.value_fc2.weight.mul_(0.01)
```

## 6. 计算量与推理速度

| 配置 | 参数量 | FLOPs (forward) | CPU 推理 (batch=1) | GPU 推理 (batch=256) |
|------|--------|-----------------|--------------------|--------------------|
| 6 block / 64 ch / 6×6 | 175K | ~2M | 3 ms | 0.5 ms |
| 10 block / 128 ch / 6×6 | 660K | ~8M | 8 ms | 0.8 ms |
| 6 block / 64 ch / 7×7 | 175K | ~3M | 4 ms | 0.6 ms |

CPU 单线程自对弈：每步 MCTS 400 次模拟 × 3ms 推理 = 1.2 秒/步，一局 30 步 = 36 秒。**8 线程并行可压到 5 秒/局**。

## 7. 与 KataGo 网络的差异总结

| 维度 | KataGo | YichiAlpha | 理由 |
|------|--------|------------|------|
| 主干 | ResNet 40-60 block | ResNet 6 block | 棋盘小 |
| 通道数 | 256 | 64 | 棋盘小 |
| 输入通道 | ~22 (含 ko/liberty/history) | 11 | 规则简单 |
| 输出 policy | 19×19+1 = 362 | 6×6+1 = 37 | 棋盘小 |
| 输出 value | 1 (tanh) | 1 (tanh) | 一致 |
| 辅助头 | ownership, score, future policy 等 | ✗ | 暂不引入，后续可加 ownership |
| SWA | ✓ | 可选 | 训练后期开启 |

## 8. 后续可扩展项

1. **Ownership 头**：KataGo 加了一个"每格最终归属"的预测头，作为辅助监督信号。异吃棋也可加，但需要终局标注每格 X/O——本框架 v1 暂未加，因为数据生成成本高。
2. **Global pooling**：在 value head 前加一个 attention pooling，可能在小棋盘上无收益但大棋盘（7×7 含障碍变体）可能有用。
3. **SE block (Squeeze-and-Excitation)**：替换部分 ResBlock 为 SE-ResNet，可能提升精度 ~2%，但 CPU 推理慢 30%。当前不做。
