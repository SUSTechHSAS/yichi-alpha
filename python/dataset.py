"""
YichiAlpha — Replay Buffer & Dataset
=====================================
FIFO buffer storing (state_tensor, policy_target, value_target) tuples.

Supports:
  - push(samples): add a list of samples
  - sample(batch_size): random sample a batch as torch tensors
  - sample_with_augmentation(): same, but applies random D4 symmetry
"""
from __future__ import annotations

import numpy as np
import torch
from typing import List, Tuple, Optional

from game import (
    GameState, state_to_tensor,
    apply_symmetry_to_planes, apply_symmetry_to_policy,
)


# A raw sample is (state, policy_vector, value_scalar)
RawSample = Tuple[GameState, np.ndarray, float]


class ReplayBuffer:
    """FIFO replay buffer with optional data augmentation."""

    def __init__(self, capacity: int = 50000, board_size: int = 6):
        self.capacity = capacity
        self.board_size = board_size
        self.buffer: List[RawSample] = []
        self.pos = 0

    def push(self, samples: List[RawSample]):
        """Add samples to the buffer (FIFO eviction)."""
        for s in samples:
            if len(self.buffer) < self.capacity:
                self.buffer.append(s)
            else:
                self.buffer[self.pos] = s
                self.pos = (self.pos + 1) % self.capacity

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random sample, return (states, policies, values) as torch tensors."""
        idx = np.random.randint(0, len(self.buffer), size=batch_size)
        states = []
        policies = []
        values = []
        for i in idx:
            s, p, v = self.buffer[i]
            states.append(state_to_tensor(s))
            policies.append(p)
            values.append([v])
        states = torch.from_numpy(np.stack(states)).float()
        policies = torch.from_numpy(np.stack(policies)).float()
        values = torch.tensor(values, dtype=torch.float32)
        return states, policies, values

    def sample_with_augmentation(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like sample(), but applies a random D4 symmetry to each sample."""
        idx = np.random.randint(0, len(self.buffer), size=batch_size)
        states = []
        policies = []
        values = []
        for i in idx:
            s, p, v = self.buffer[i]
            planes = state_to_tensor(s)
            # Random symmetry
            k = np.random.randint(4)
            flip = np.random.rand() < 0.5
            planes = apply_symmetry_to_planes(planes, k, flip)
            p = apply_symmetry_to_policy(p, self.board_size, k, flip)
            states.append(planes)
            policies.append(p)
            values.append([v])
        states = torch.from_numpy(np.stack(states)).float()
        policies = torch.from_numpy(np.stack(policies)).float()
        values = torch.tensor(values, dtype=torch.float32)
        return states, policies, values


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    buf = ReplayBuffer(capacity=100, board_size=6)

    # Generate fake samples
    for _ in range(50):
        s = GameState.initial()
        p = np.random.dirichlet([1.0] * 37)
        v = np.random.uniform(-1, 1)
        buf.push([(s, p, v)])

    print(f"Buffer size: {len(buf)}")
    states, policies, values = buf.sample(8)
    print(f"States: {states.shape}")
    print(f"Policies: {policies.shape}, sum={policies.sum(dim=-1)}")
    print(f"Values: {values.shape}, range=[{values.min():.3f}, {values.max():.3f}]")

    # Augmented
    states_aug, policies_aug, values_aug = buf.sample_with_augmentation(8)
    print(f"\nAugmented:")
    print(f"States: {states_aug.shape}")
    print(f"Policies: {policies_aug.shape}, sum={policies_aug.sum(dim=-1)}")
