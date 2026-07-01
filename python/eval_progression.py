"""
Evaluate model progression: compare early vs later checkpoints.
Shows that training is actually learning (later iter beats earlier iter).
"""
import sys
sys.path.insert(0, '/home/z/my-project/download/yichi-alpha/python')

import time
import numpy as np
import torch
from pathlib import Path

from game import GameState, GameConfig, X, O
from model import load_checkpoint
from evaluate import MCTSAgent, RandomAgent, play_match


def arena_compare(model_a_path, model_b_path, n_games=10, mcts_sims=50,
                  board_size=6, device='cpu', label_a='A', label_b='B'):
    """Play n_games between model_a (X) and model_b (O), then swap colors."""
    model_a = load_checkpoint(model_a_path, device=device)
    model_b = load_checkpoint(model_b_path, device=device)

    cfg = GameConfig(board_size=board_size)
    rng = np.random.default_rng(42)

    agent_a = MCTSAgent(model_a, n_simulations=mcts_sims, temperature=0.0, device=device)
    agent_b = MCTSAgent(model_b, n_simulations=mcts_sims, temperature=0.0, device=device)

    wins_a = 0
    wins_b = 0
    draws = 0

    print(f"Arena: {label_a} ({Path(model_a_path).name}) vs {label_b} ({Path(model_b_path).name})")
    print(f"  {n_games} games, {mcts_sims} MCTS sims each, alternating colors\n")

    t0 = time.time()
    for g in range(n_games):
        if g % 2 == 0:
            # A plays X, B plays O
            w = play_match(agent_a, agent_b, cfg)
            if w == X: wins_a += 1
            elif w == O: wins_b += 1
            else: draws += 1
        else:
            # B plays X, A plays O
            w = play_match(agent_b, agent_a, cfg)
            if w == O: wins_a += 1
            elif w == X: wins_b += 1
            else: draws += 1
        if (g + 1) % 2 == 0:
            print(f"  After {g+1} games: {label_a}={wins_a}, {label_b}={wins_b}, draws={draws}")

    dt = time.time() - t0
    print(f"\nFinal: {label_a}={wins_a}, {label_b}={wins_b}, draws={draws} ({dt:.1f}s)")
    print(f"  {label_a} win rate: {wins_a/n_games:.1%}")
    print(f"  {label_b} win rate: {wins_b/n_games:.1%}")
    return wins_a, wins_b, draws


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='../checkpoints')
    parser.add_argument('--games', type=int, default=6)
    parser.add_argument('--mcts_sims', type=int, default=40)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    # Find all iter checkpoints
    iters = sorted(ckpt_dir.glob("model_iter*.pt"),
                   key=lambda p: int(p.stem.replace('model_iter', '')))
    if len(iters) < 2:
        print(f"Need at least 2 checkpoints, found {len(iters)}")
        return

    print(f"Found {len(iters)} checkpoints: {[p.name for p in iters]}")
    print()

    # Compare first vs last
    first = iters[0]
    last = iters[-1]
    arena_compare(str(first), str(last),
                  n_games=args.games, mcts_sims=args.mcts_sims,
                  device=args.device,
                  label_a=f'iter{int(first.stem.replace("model_iter",""))}',
                  label_b=f'iter{int(last.stem.replace("model_iter",""))}')

    # If more than 2, also compare middle vs last
    if len(iters) > 2:
        mid = iters[len(iters)//2]
        print()
        arena_compare(str(mid), str(last),
                      n_games=args.games, mcts_sims=args.mcts_sims,
                      device=args.device,
                      label_a=f'iter{int(mid.stem.replace("model_iter",""))}',
                      label_b=f'iter{int(last.stem.replace("model_iter",""))}')


if __name__ == '__main__':
    main()
