"""
play.py — 人机对战：你 vs 训练好的 AI
==========================================

在终端里和训练好的模型下异吃棋。

用法:
    python play.py                          # 默认：你执 X（先手），AI 执 O
    python play.py --side o                 # 你执 O（后手），AI 执 X（先手）
    python play.py --sims 200               # AI 用更多 MCTS 模拟（更强但更慢）
    python play.py --random                 # AI 用随机策略（不用模型，最弱）
    python play.py --checkpoint ../checkpoints/model_iter3.pt

操作:
    - 棋盘坐标：列 A-F (或 A-G)，行 1-6 (或 1-7)
    - 输入如 "C3" 表示在第 C 列第 3 行落子
    - 输入 "quit" 或 "q" 退出
    - 输入 "moves" 显示所有合法落子点
    - 输入 "undo" 悔棋一步（双方都退）
"""
from __future__ import annotations

import argparse
import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from game import (
    GameState, GameConfig, X, O, BLOCK, EMPTY,
    state_to_tensor, CELL_CHARS,
)
from model import load_checkpoint
from mcts import MCTS
from evaluate import RandomAgent


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
PLAYER_LABEL = {X: 'X (橙)', O: 'O (蓝)'}
PLAYER_SHORT = {X: 'X', O: 'O'}


def render_board(state: GameState, last_move=None, highlight_legal=False):
    """Render the board as ASCII art with coordinates."""
    n = state.board_size
    print()
    # Column header
    print("    " + "  ".join(chr(ord('A') + c) for c in range(n)) + "  ")
    print("   +" + "---+" * n)

    legal_set = set(state.legal_moves()) if highlight_legal else set()

    for r in range(n):
        row_label = f"{r+1:>2}"
        cells = []
        for c in range(n):
            t = int(state.types[r, c])
            h = int(state.health[r, c])
            ch = CELL_CHARS[t]
            if t in (X, O):
                cell = f"{ch}{h}"
            elif t == BLOCK:
                cell = " # "
            else:
                # Empty cell
                if (r, c) in legal_set:
                    coord = f"{chr(ord('A') + c)}{r+1}"
                    cell = f"·{coord[-1]}·"[:3] if len(coord) == 2 else " · "
                    # Simpler: just show a dot
                    cell = " · "
                else:
                    cell = " · "
            # Highlight last move with brackets
            if last_move and (r, c) == last_move:
                cell = f"[{ch}{h}]" if t in (X, O) else f"[{cell.strip()}]"
                # Actually just mark with parens for clarity
                cell = f"({ch}{h})" if t in (X, O) else cell
            cells.append(cell)
        print(f"{row_label} |" + "|".join(cells) + "|")
        print("   +" + "---+" * n)

    print()
    print(f"  当前轮到: {PLAYER_LABEL[state.current_player]}    步数: {state.step}")
    x_count = int((state.types == X).sum())
    o_count = int((state.types == O).sum())
    print(f"  棋子数: X={x_count}  O={o_count}")
    print()


def parse_move(input_str: str, board_size: int):
    """Parse user input like 'C3' or 'c3' into (row, col). Returns None if invalid."""
    s = input_str.strip().upper()
    if len(s) < 2:
        return None
    col_char = s[0]
    row_str = s[1:]
    if not col_char.isalpha() or not row_str.isdigit():
        return None
    col = ord(col_char) - ord('A')
    try:
        row = int(row_str) - 1
    except ValueError:
        return None
    if not (0 <= col < board_size and 0 <= row < board_size):
        return None
    return (row, col)


def show_legal_moves(state: GameState):
    """Print all legal moves with coordinates."""
    legal = state.legal_moves()
    print(f"\n合法落子点 ({len(legal)} 个):")
    # Group by row for readability
    coords = [f"{chr(ord('A') + c)}{r+1}" for r, c in legal]
    # Print 8 per line
    for i in range(0, len(coords), 8):
        print("  " + "  ".join(coords[i:i+8]))
    print()


# ---------------------------------------------------------------------------
# AI Agent wrapper
# ---------------------------------------------------------------------------
class AIAgent:
    """MCTS + neural network agent."""
    def __init__(self, model, n_simulations=100, device='cpu', show_thinking=True):
        self.model = model
        self.mcts = MCTS(model, n_simulations=n_simulations, device=device)
        self.show_thinking = show_thinking

    def select_move(self, state: GameState):
        if self.show_thinking:
            print(f"  AI 思考中... (MCTS {self.mcts.n_simulations} 次模拟)")
        t0 = time.time()
        root = self.mcts.search(state, add_noise=False)
        pi = self.mcts.get_action_distribution(root, temperature=0.0)
        dt = time.time() - t0

        # Show top 3 candidate moves
        if self.show_thinking:
            n = state.board_size
            top3_idx = np.argsort(pi)[::-1][:3]
            print(f"  AI 候选落子 (思考 {dt:.1f}s):")
            for idx in top3_idx:
                if idx < n * n and pi[idx] > 0:
                    r, c = idx // n, idx % n
                    coord = f"{chr(ord('A') + c)}{r+1}"
                    print(f"    {coord}: {pi[idx]*100:.1f}% 访问次数")
            # Also show value estimate
            x = state_to_tensor(state)
            _, value = self.model.predict(x)
            print(f"  AI 评估当前局面 (自己视角): {value[0]:+.2f}  (>0 = AI 觉得自己优势)")

        move = self.mcts.select_action(root, temperature=0.0)
        return move


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------
def play_game(args):
    """Run one human-vs-AI game."""
    cfg = GameConfig(board_size=args.board_size)
    state = GameState.initial(cfg)

    human_side = X if args.side.lower() in ('x', '1') else O
    ai_side = O if human_side == X else X

    print("=" * 50)
    print(f"  异吃棋 — 人机对战")
    print("=" * 50)
    print(f"  你:    {PLAYER_LABEL[human_side]}")
    print(f"  AI:    {PLAYER_LABEL[ai_side]}")
    print(f"  棋盘:  {args.board_size}×{args.board_size}")
    if args.random:
        print(f"  AI 策略: 随机 (最弱)")
    else:
        print(f"  AI 策略: MCTS ({args.sims} 次模拟) + {args.checkpoint}")
    print()
    print("  坐标输入：列字母 + 行号，如 'C3'")
    print("  输入 'moves' 查看合法落子点")
    print("  输入 'quit' 退出")
    print("  输入 'undo' 悔棋一步")
    print("=" * 50)

    # Set up AI
    if args.random:
        ai_agent = RandomAgent()
    else:
        model = load_checkpoint(args.checkpoint, device='cpu')
        ai_agent = AIAgent(model, n_simulations=args.sims, device='cpu',
                          show_thinking=args.show_thinking)

    history = []  # for undo

    # Game loop
    while not state.is_terminal():
        render_board(state, highlight_legal=False)

        current = state.current_player
        if current == human_side:
            # Human turn
            while True:
                try:
                    user_input = input(f"  你的回合 ({PLAYER_SHORT[human_side]}) — 输入坐标 (如 C3): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n退出游戏。")
                    return
                if user_input.lower() in ('quit', 'q', 'exit'):
                    print("退出游戏。")
                    return
                if user_input.lower() == 'moves':
                    show_legal_moves(state)
                    continue
                if user_input.lower() == 'undo':
                    if len(history) >= 2:
                        # Undo both human's and AI's last move
                        history.pop()  # AI's move
                        prev = history.pop()
                        state = prev
                        print("  已悔棋一步（双方）。\n")
                    elif len(history) == 1:
                        prev = history.pop()
                        state = prev
                        print("  已悔棋一步。\n")
                    else:
                        print("  没有可悔的棋。\n")
                    break
                if not user_input:
                    continue
                move = parse_move(user_input, args.board_size)
                if move is None:
                    print(f"  无效输入: '{user_input}'。请输入如 'C3' 的坐标。")
                    continue
                if move not in state.legal_moves():
                    print(f"  ({chr(ord('A') + move[1])}{move[0]+1}) 不是合法落子点。")
                    print(f"  该位置可能已被占用，或不在已有棋子 2 格范围内。")
                    continue
                # Apply move
                history.append(state.clone())
                state.apply_move(move)
                print(f"\n  你落子: {chr(ord('A') + move[1])}{move[0]+1}")
                break
        else:
            # AI turn
            print(f"\n  AI 回合 ({PLAYER_SHORT[ai_side]})...")
            move = ai_agent.select_move(state)
            if move is None:
                print("  AI 无合法落子，游戏结束。")
                break
            history.append(state.clone())
            state.apply_move(move)
            print(f"  AI 落子: {chr(ord('A') + move[1])}{move[0]+1}")

    # Game over
    render_board(state)
    x_count = int((state.types == X).sum())
    o_count = int((state.types == O).sum())
    winner = state.winner()

    print("=" * 50)
    print(f"  游戏结束!")
    print(f"  X={x_count}  O={o_count}")
    if winner == -1:
        print(f"  结果: 平局")
    elif winner == human_side:
        print(f"  结果: 你赢了! 🎉")
    else:
        print(f"  结果: AI 赢了 🤖")
    print("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='人机对战: 你 vs 训练好的 AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python play.py                              # 你执 X 先手，AI 用默认模型
  python play.py --side o                     # 你执 O 后手
  python play.py --sims 200                   # AI 更强 (更多 MCTS 模拟)
  python play.py --random                     # AI 用随机策略 (最弱，适合新手)
  python play.py --checkpoint model_iter3.pt  # 指定模型
        """
    )
    parser.add_argument('--checkpoint', type=str,
                        default='../checkpoints/model_iter3.pt',
                        help='模型 checkpoint 路径')
    parser.add_argument('--side', type=str, default='x',
                        choices=['x', 'o', 'X', 'O', '1', '2'],
                        help='你执哪一方 (x=先手, o=后手)')
    parser.add_argument('--sims', type=int, default=80,
                        help='AI 的 MCTS 模拟次数 (越多越强越慢)')
    parser.add_argument('--board_size', type=int, default=6,
                        choices=[5, 6, 7],
                        help='棋盘大小')
    parser.add_argument('--random', action='store_true',
                        help='AI 用随机策略 (不用模型)')
    parser.add_argument('--no-thinking', action='store_true',
                        help='不显示 AI 思考过程')
    args = parser.parse_args()

    args.show_thinking = not args.no_thinking

    if not args.random and not Path(args.checkpoint).exists():
        print(f"错误: 模型文件不存在: {args.checkpoint}")
        print("请先运行训练: python train.py --config ../configs/quick.yaml --iterations 3")
        print("或用随机策略: python play.py --random")
        sys.exit(1)

    try:
        play_game(args)
    except KeyboardInterrupt:
        print("\n\n游戏中断。")


if __name__ == '__main__':
    main()
