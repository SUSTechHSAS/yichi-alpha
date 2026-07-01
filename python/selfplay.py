"""
YichiAlpha — Self-Play Data Generation
=======================================
Run self-play games with the current model to generate training samples.

Each game produces a list of (state, mcts_policy, terminal_value) samples.
The terminal value is backfilled from the game result, perspective-aligned
to the player who was about to move at that state.
"""
from __future__ import annotations

import time
import numpy as np
import torch
from typing import List, Tuple
from dataclasses import dataclass

from game import GameState, GameConfig, X, O
from mcts import MCTS
from model import YichiNet
from dataset import RawSample


def temperature_schedule(step: int, total_steps_est: int) -> float:
    """High temperature early (explore), low late (exploit)."""
    if total_steps_est <= 0:
        return 1.0
    ratio = step / total_steps_est
    if ratio < 0.3:
        return 1.0
    elif ratio < 0.7:
        return 0.5
    else:
        return 0.0


def self_play_game(
    model: YichiNet,
    mcts: MCTS,
    config: GameConfig,
    rng: Optional[np.random.Generator] = None,
    max_steps: int = 100,
) -> List[RawSample]:
    """Run one self-play game, return list of training samples.

    Parameters
    ----------
    model : YichiNet
    mcts : MCTS
        Configured MCTS searcher (will be used as-is).
    config : GameConfig
    rng : random generator for reproducibility
    max_steps : safety cap to avoid infinite loops
    """
    if rng is None:
        rng = np.random.default_rng()

    state = GameState.initial(config)
    trajectory: List[Tuple[GameState, np.ndarray, int]] = []
    # Each entry: (state_snapshot, mcts_policy, player_who_moved)

    step = 0
    while not state.is_terminal() and step < max_steps:
        # Estimate remaining steps for temperature scheduling
        # Rough: total cells - current filled
        n_cells = config.board_size ** 2
        filled = int((state.types != 0).sum())  # non-empty cells
        est_total = max(n_cells, filled + 1)
        temperature = temperature_schedule(filled, est_total)

        # MCTS search
        root = mcts.search(state, add_noise=True)
        pi = mcts.get_action_distribution(root, temperature=temperature)

        # Snapshot state BEFORE applying move (this is what the network sees)
        snapshot = state.clone()

        # Sample action
        if temperature == 0:
            idx = int(np.argmax(pi))
        else:
            idx = int(rng.choice(len(pi), p=pi))

        n = config.board_size
        if idx >= n * n:
            # Pass — should only happen if no legal moves (very rare)
            legal = state.legal_moves()
            if not legal:
                break
            move = legal[0]
        else:
            move = (idx // n, idx % n)

        # Record sample (state snapshot, policy, player who moved)
        trajectory.append((snapshot, pi, state.current_player))

        # Apply move
        state.apply_move(move)
        step += 1

    # Backfill terminal value
    winner = state.winner()  # X, O, or -1 (draw)
    samples: List[RawSample] = []
    for snap, pi, player in trajectory:
        if winner == -1:
            z = 0.0
        elif winner == player:
            z = +1.0
        else:
            z = -1.0
        samples.append((snap, pi, z))

    return samples


def self_play_random_baseline(
    config: GameConfig,
    rng: Optional[np.random.Generator] = None,
    max_steps: int = 100,
) -> List[RawSample]:
    """Generate a self-play game using random play (for cold-start).

    Returns the same format as self_play_game but with uniform policy.
    """
    if rng is None:
        rng = np.random.default_rng()

    state = GameState.initial(config)
    trajectory = []
    step = 0
    while not state.is_terminal() and step < max_steps:
        legal = state.legal_moves()
        if not legal:
            break
        # Uniform policy over legal moves
        n = config.board_size
        pi = np.zeros(n * n + 1, dtype=np.float32)
        for m in legal:
            pi[m[0] * n + m[1]] = 1.0
        pi /= pi.sum()

        snapshot = state.clone()
        player = state.current_player
        move = legal[rng.integers(len(legal))]
        trajectory.append((snapshot, pi, player))
        state.apply_move(move)
        step += 1

    winner = state.winner()
    samples = []
    for snap, pi, player in trajectory:
        if winner == -1:
            z = 0.0
        elif winner == player:
            z = +1.0
        else:
            z = -1.0
        samples.append((snap, pi, z))
    return samples


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time
    from typing import Optional

    print("=== Self-Play Test ===")
    model = YichiNet(board_size=6)
    mcts = MCTS(model, n_simulations=30, device='cpu')

    t0 = time.time()
    rng = np.random.default_rng(42)
    samples = self_play_game(model, mcts, GameConfig(), rng=rng, max_steps=50)
    elapsed = time.time() - t0

    print(f"Game generated {len(samples)} samples in {elapsed:.1f}s")
    print(f"Sample 0: state step={samples[0][0].step}, policy sum={samples[0][1].sum():.3f}, value={samples[0][2]}")
    print(f"Sample -1: state step={samples[-1][0].step}, policy sum={samples[-1][1].sum():.3f}, value={samples[-1][2]}")
    print(f"Final board:")
    print(samples[-1][0])
    print(f"Winner (from last sample's value): {samples[-1][2]}")
