"""
YichiAlpha — MCTS with Neural Network guidance
================================================
Monte Carlo Tree Search adapted for 异吃棋 chain reactions.

Key design (see docs/04_mcts_adaptation.md):
  - Transition = (move + full chain refresh until stable)
  - value = current-player perspective
  - backup with perspective flip
  - Dirichlet noise at root for exploration
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

from game import (
    GameState, GameConfig, X, O, BLOCK, EMPTY,
    state_to_tensor, move_to_index, index_to_move,
    N_FEATURE_CHANNELS,
)
from model import YichiNet


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------
@dataclass
class MCTSNode:
    """A node in the MCTS search tree."""
    state: GameState
    parent: Optional['MCTSNode'] = None
    move: Optional[Tuple[int, int]] = None    # move that led here (None for root)
    prior: float = 0.0                         # P(s, a) from network
    children: Dict[Tuple[int, int], 'MCTSNode'] = field(default_factory=dict)
    N: int = 0
    W: float = 0.0
    Q: float = 0.0
    is_expanded: bool = False

    def ucb_score(self, c_puct: float) -> float:
        """PUCT: Q + c_puct * P * sqrt(N_parent) / (1 + N)

        Note: self.Q is stored from "the player to move at this node's
        perspective". When the parent is choosing among children, the
        parent's player is the OPPONENT of this node's player, so the
        parent's value for moving to this child is -self.Q.
        """
        if self.parent is None:
            return 0.0
        u = c_puct * self.prior * (math.sqrt(self.parent.N) / (1 + self.N))
        return -self.Q + u  # negate: parent wants to MAXIMIZE its own value

    def best_child(self, c_puct: float) -> 'MCTSNode':
        return max(self.children.values(), key=lambda c: c.ucb_score(c_puct))


# ---------------------------------------------------------------------------
# MCTS Searcher
# ---------------------------------------------------------------------------
class MCTS:
    """AlphaZero-style MCTS with NN guidance.

    Parameters
    ----------
    model : YichiNet
        Policy-Value network.
    c_puct : float
        Exploration constant. Default 1.5 (higher than AlphaZero's 1.0 because
        异吃棋 value signal is weaker).
    n_simulations : int
        Number of MCTS simulations per move.
    dirichlet_alpha : float
        Dirichlet noise concentration. 0.3 for ~37 actions (similar to chess).
    dirichlet_epsilon : float
        Noise mixing weight (0.25 standard).
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(self, model: YichiNet, c_puct: float = 1.5,
                 n_simulations: int = 200,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 device: str = 'cpu'):
        self.model = model
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = torch.device(device)
        self.model.to(self.device)

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------
    def search(self, root_state: GameState, add_noise: bool = True) -> MCTSNode:
        """Run MCTS from root_state, return the root node (with stats)."""
        root = MCTSNode(state=root_state.clone())

        # Evaluate root
        policy, value = self._evaluate(root.state)
        self._expand(root, policy, add_noise=add_noise)

        # If root is terminal (rare — only if board is full), return immediately
        if root.state.is_terminal():
            return root

        for _ in range(self.n_simulations):
            node = self._select(root)
            v = self._evaluate_and_expand(node)
            self._backup(node, v)

        return root

    def get_action_distribution(self, root: MCTSNode,
                                 temperature: float = 1.0) -> np.ndarray:
        """Return policy target π ~ visits^(1/T) over action space (N*N+1,)."""
        n = root.state.board_size
        visits = np.zeros(n * n + 1, dtype=np.float32)
        for move, child in root.children.items():
            idx = move_to_index(move, n)
            visits[idx] = child.N

        if temperature == 0:
            # Greedy
            result = np.zeros_like(visits)
            result[visits.argmax()] = 1.0
            return result

        visits = visits ** (1.0 / temperature)
        total = visits.sum()
        if total == 0:
            # Fallback: uniform
            return np.ones_like(visits) / len(visits)
        return visits / total

    def select_action(self, root: MCTSNode, temperature: float = 1.0,
                       rng: Optional[np.random.Generator] = None) -> Tuple[int, int]:
        """Sample an action from the MCTS visit distribution."""
        pi = self.get_action_distribution(root, temperature)
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(len(pi), p=pi)
        # Pass action (idx == N*N) — extremely rare, only when no legal moves
        if idx == root.state.board_size ** 2:
            # Shouldn't happen because apply_move requires legal move
            # Fallback: pick first legal
            legal = root.state.legal_moves()
            return legal[0] if legal else (0, 0)
        return index_to_move(idx, root.state.board_size)

    # ----------------------------------------------------------------
    # Internal: selection, evaluation, expansion, backup
    # ----------------------------------------------------------------
    def _select(self, root: MCTSNode) -> MCTSNode:
        """Traverse tree by PUCT until leaf."""
        node = root
        while node.is_expanded and not node.state.is_terminal():
            node = node.best_child(self.c_puct)
        return node

    def _evaluate_and_expand(self, node: MCTSNode) -> float:
        """Evaluate leaf with network, expand, return value (node's perspective)."""
        if node.state.is_terminal():
            return node.state.terminal_reward_for_current_player()

        policy, value = self._evaluate(node.state)
        self._expand(node, policy, add_noise=False)
        return float(value)

    def _evaluate(self, state: GameState) -> Tuple[np.ndarray, float]:
        """Run network on state, return (policy_probs, value_scalar)."""
        x = state_to_tensor(state)
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(x)
            policy = F.softmax(logits, dim=-1)
        return policy.squeeze(0).cpu().numpy(), float(value.squeeze().item())

    def _expand(self, node: MCTSNode, policy: np.ndarray, add_noise: bool):
        """Create child nodes for all legal moves."""
        legal_moves = node.state.legal_moves()
        n = node.state.board_size

        if add_noise and len(legal_moves) > 0:
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * len(legal_moves)
            )
        else:
            noise = np.zeros(len(legal_moves))

        # Normalize priors over legal moves only
        priors = np.array([policy[move_to_index(m, n)] for m in legal_moves])
        if add_noise:
            priors = (1 - self.dirichlet_epsilon) * priors + self.dirichlet_epsilon * noise
        # Renormalize
        total = priors.sum()
        if total > 0:
            priors = priors / total
        else:
            # Uniform fallback
            priors = np.ones(len(legal_moves)) / len(legal_moves)

        for move, p in zip(legal_moves, priors):
            child_state = node.state.clone()
            child_state.apply_move(move)
            node.children[move] = MCTSNode(
                state=child_state, parent=node, move=move, prior=float(p)
            )
        node.is_expanded = True

    def _backup(self, node: MCTSNode, value: float):
        """Backup value with perspective flip."""
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value  # flip perspective
            node = node.parent


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    print("=== MCTS Self-test ===")
    model = YichiNet(board_size=6)
    mcts = MCTS(model, n_simulations=50, device='cpu')

    state = GameState.initial()
    print("Initial state:")
    print(state)
    print()

    t0 = time.time()
    root = mcts.search(state, add_noise=True)
    elapsed = time.time() - t0
    print(f"Search done in {elapsed:.2f}s ({mcts.n_simulations} sims)")

    pi = mcts.get_action_distribution(root, temperature=1.0)
    print(f"Action distribution sum: {pi.sum():.4f}")
    print(f"Top 5 actions:")
    n = state.board_size
    top5 = np.argsort(pi)[::-1][:5]
    for idx in top5:
        if idx < n * n:
            r, c = index_to_move(idx, n)
            print(f"  ({r},{c}): {pi[idx]:.3f}")
        else:
            print(f"  pass: {pi[idx]:.3f}")

    # Pick an action
    move = mcts.select_action(root, temperature=1.0)
    print(f"\nSelected move: {move}")
    state.apply_move(move)
    print("After move:")
    print(state)
