"""
YichiAlpha — Evaluation
========================
Evaluate a trained model against baselines:
  - random play
  - greedy (1-ply lookahead with learned value)
  - previous best model (Arena)
"""
from __future__ import annotations

import argparse
import sys
import time
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from game import GameState, GameConfig, X, O
from model import YichiNet, load_checkpoint
from mcts import MCTS


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
class RandomAgent:
    """Uniformly random legal move."""
    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
        self.name = "random"

    def select_move(self, state: GameState) -> Optional[Tuple[int, int]]:
        legal = state.legal_moves()
        if not legal:
            return None
        return legal[int(self.rng.integers(len(legal)))]


class MCTSAgent:
    """Use MCTS with the given model to select moves."""
    def __init__(self, model: YichiNet, n_simulations: int = 100,
                 c_puct: float = 1.5, temperature: float = 0.0,
                 device: str = 'cpu'):
        self.model = model
        self.mcts = MCTS(model, c_puct=c_puct, n_simulations=n_simulations,
                         device=device)
        self.temperature = temperature
        self.name = f"mcts{n_simulations}"

    def select_move(self, state: GameState) -> Optional[Tuple[int, int]]:
        if state.is_terminal():
            return None
        root = self.mcts.search(state, add_noise=False)
        pi = self.mcts.get_action_distribution(root, temperature=self.temperature)
        if self.temperature == 0:
            idx = int(np.argmax(pi))
        else:
            idx = int(np.random.choice(len(pi), p=pi))
        n = state.board_size
        if idx >= n * n:
            legal = state.legal_moves()
            return legal[0] if legal else None
        return (idx // n, idx % n)


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------
def play_match(agent_x, agent_o, game_config: GameConfig,
               max_steps: int = 100) -> int:
    """Play one game, return winner (X=1, O=2, -1=draw)."""
    state = GameState.initial(game_config)
    step = 0
    while not state.is_terminal() and step < max_steps:
        agent = agent_x if state.current_player == X else agent_o
        move = agent.select_move(state)
        if move is None:
            break
        state.apply_move(move)
        step += 1
    return state.winner()


def evaluate_against_random(model: YichiNet, n_games: int = 20,
                             mcts_sims: int = 100,
                             game_config: Optional[GameConfig] = None,
                             device: str = 'cpu') -> dict:
    """Evaluate model vs random agent. Model plays both X and O equally."""
    game_config = game_config or GameConfig()
    rng = np.random.default_rng()
    random_agent = RandomAgent(rng=rng)
    model_agent = MCTSAgent(model, n_simulations=mcts_sims, temperature=0.0, device=device)

    wins = 0
    losses = 0
    draws = 0
    for g in range(n_games):
        if g % 2 == 0:
            # Model plays X
            w = play_match(model_agent, random_agent, game_config)
            if w == X: wins += 1
            elif w == O: losses += 1
            else: draws += 1
        else:
            # Model plays O
            w = play_match(random_agent, model_agent, game_config)
            if w == O: wins += 1
            elif w == X: losses += 1
            else: draws += 1

    result = {
        "n_games": n_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_games,
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="YichiAlpha evaluation")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model .pt file')
    parser.add_argument('--games', type=int, default=20,
                        help='Number of evaluation games')
    parser.add_argument('--mcts_sims', type=int, default=100,
                        help='MCTS simulations per move')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--board_size', type=int, default=6)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, device=args.device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    game_config = GameConfig(board_size=args.board_size)
    print(f"\nEvaluating vs random agent ({args.games} games, MCTS {args.mcts_sims} sims)...")
    t0 = time.time()
    result = evaluate_against_random(
        model, n_games=args.games,
        mcts_sims=args.mcts_sims,
        game_config=game_config,
        device=args.device,
    )
    dt = time.time() - t0
    print(f"\nResults ({dt:.1f}s):")
    print(f"  Wins:   {result['wins']}/{result['n_games']} ({result['win_rate']:.1%})")
    print(f"  Losses: {result['losses']}/{result['n_games']}")
    print(f"  Draws:  {result['draws']}/{result['n_games']}")


if __name__ == '__main__':
    main()
