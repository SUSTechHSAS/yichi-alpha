"""Debug: see what the trained model predicts in a sample game."""
import sys
sys.path.insert(0, '/home/z/my-project/download/yichi-alpha/python')

import numpy as np
import torch
from game import GameState, GameConfig, X, O, state_to_tensor, index_to_move
from model import load_checkpoint
from mcts import MCTS

model = load_checkpoint('/home/z/my-project/download/yichi-alpha/checkpoints/model_iter3.pt')
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

state = GameState.initial()
print("Initial state:")
print(state)

# Network direct prediction (no MCTS)
planes = state_to_tensor(state)
policy, value = model.predict(planes)
print(f"\nDirect network prediction:")
print(f"  Value (current player X): {value[0]:.3f}")
print(f"  Top-5 policy actions:")
n = state.board_size
top5 = np.argsort(policy)[::-1][:5]
for idx in top5:
    if idx < n*n:
        r, c = index_to_move(idx, n)
        print(f"    ({r},{c}): {policy[idx]:.4f}")
    else:
        print(f"    pass: {policy[idx]:.4f}")

# MCTS prediction
mcts = MCTS(model, n_simulations=30, device='cpu')
root = mcts.search(state, add_noise=False)
pi = mcts.get_action_distribution(root, temperature=0.0)
print(f"\nMCTS (T=0, greedy) action distribution:")
print(f"  Top-5 actions:")
top5 = np.argsort(pi)[::-1][:5]
for idx in top5:
    if idx < n*n:
        r, c = index_to_move(idx, n)
        print(f"    ({r},{c}): {pi[idx]:.4f}")
    else:
        print(f"    pass: {pi[idx]:.4f}")

# After 1 move at center (3,3) - simple position
state2 = GameState.initial()
state2.apply_move((2, 2))  # X plays center
state2.apply_move((2, 3))  # O plays adjacent
print(f"\nAfter X(2,2) O(2,3):")
print(state2)
print(f"Current player: {'X' if state2.current_player==X else 'O'}")

planes2 = state_to_tensor(state2)
policy2, value2 = model.predict(planes2)
print(f"\nNetwork prediction (perspective={state2.current_player}):")
print(f"  Value: {value2[0]:.3f}")
print(f"  Top-5 policy:")
top5 = np.argsort(policy2)[::-1][:5]
for idx in top5:
    if idx < n*n:
        r, c = index_to_move(idx, n)
        print(f"    ({r},{c}): {policy2[idx]:.4f}")

# Check policy entropy
ent = -(policy * np.log(policy + 1e-12)).sum()
print(f"\nPolicy entropy (initial state): {ent:.3f} (max for 37 actions = {np.log(37):.3f})")
