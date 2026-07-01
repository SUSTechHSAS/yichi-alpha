"""
YichiAlpha — Policy-Value Network (PyTorch)
============================================
ResNet backbone + policy/value heads, designed for the 异吃棋 game.

Architecture (see docs/03_network_architecture.md):
    Input: (B, 11, N, N)
      │
      ├─ Stem: Conv3x3(11→64) + BN + ReLU
      ├─ ResBlock × 6
      ├─ Policy head: Conv1x1(64→2) + BN + ReLU + Linear(2N² → N²+1)
      └─ Value head:  Conv1x1(64→1) + BN + ReLU + Linear(N²→32) + ReLU + Linear(32→1) + Tanh
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


# ---------------------------------------------------------------------------
# Policy-Value Network
# ---------------------------------------------------------------------------
class YichiNet(nn.Module):
    """Policy-Value network for 异吃棋.

    Input:  (B, 11, N, N)  float32, perspective-aligned
    Output: policy_logits (B, N*N+1)
            value         (B, 1)  in [-1, +1]
    """

    def __init__(self, board_size: int = 6, in_channels: int = 11,
                 channels: int = 64, n_blocks: int = 6):
        super().__init__()
        self.board_size = board_size
        self.in_channels = in_channels
        self.channels = channels
        self.n_blocks = n_blocks
        self.n_actions = board_size * board_size + 1  # +1 for pass

        # Stem
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

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Shrink final layers so initial policy is near-uniform, value near 0
        with torch.no_grad():
            self.policy_fc.weight.mul_(0.01)
            self.value_fc2.weight.mul_(0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.blocks(h)

        # Policy
        p = self.policy_conv(h)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)  # logits, NOT softmaxed

        # Value
        v = self.value_conv(h)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def predict(self, x, device: torch.device = None):
        """Single-sample inference, returns numpy arrays.

        Accepts numpy array or torch tensor.
        """
        if device is None:
            device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = x.to(device)
            logits, value = self.forward(x)
            policy = F.softmax(logits, dim=-1)
        if was_training:
            self.train()
        return policy.squeeze(0).cpu().numpy(), value.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------
def save_checkpoint(model: YichiNet, path: str, extra: dict = None):
    """Save model weights + config."""
    state = {
        'state_dict': model.state_dict(),
        'board_size': model.board_size,
        'in_channels': model.in_channels,
        'channels': model.channels,
        'n_blocks': model.n_blocks,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(path: str, device: str = 'cpu') -> YichiNet:
    """Load model from checkpoint."""
    state = torch.load(path, map_location=device, weights_only=False)
    model = YichiNet(
        board_size=state['board_size'],
        in_channels=state['in_channels'],
        channels=state['channels'],
        n_blocks=state['n_blocks'],
    ).to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    model = YichiNet(board_size=6)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"YichiNet: {n_params:,} parameters ({n_params/1024:.1f}K)")

    # Forward pass test
    x = torch.randn(4, 11, 6, 6)
    policy, value = model(x)
    print(f"Input:  {x.shape}")
    print(f"Policy: {policy.shape}  (expect (4, 37))")
    print(f"Value:  {value.shape}  (expect (4, 1), range [-1,1])")
    print(f"Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")
    print(f"Policy sum (after softmax): {F.softmax(policy, dim=-1).sum(dim=-1)}")
