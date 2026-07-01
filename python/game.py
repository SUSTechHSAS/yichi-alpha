"""
YichiAlpha — Game Engine (Python)
==================================
Pure-Python/NumPy implementation of 异吃棋 (Different-Eating Chess).

This module is the canonical reference implementation of the game rules.
The C++ engine in `engine/src/rules.cpp` must produce identical results.

Game summary
------------
- N×N board (default 6×6), two players X (first) and O (second).
- Each piece has a health value (default 2).
- Players alternate placing pieces on empty cells within 2-Chebyshev distance
  of any existing piece.
- After each placement, a chain reaction `refresh_board()` runs until stable:
    1. block_rule(): cell becomes BLOCK if it has ≥2 X and ≥2 O direct neighbors
    2. heal_rule() for all pieces: reset health, then if same-color direct
       neighbors n≥2, health = h0 + (n+m)*healpower - 1 (m = diagonal same-color)
    3. damage_rule() + death_rule() for current player's pieces:
       if opposite-color direct neighbors n≥2, health -= (n+m)*attackpower
       if health ≤ 0, flip to opponent color, health = 2, mark changed
    4. heal_rule() for all again
    5. damage_rule() + death_rule() for opponent's pieces
    6. Loop back to 1 if any death_rule triggered a flip.
- Game ends when board is full. Player with more pieces wins.

Reference: https://sustechhsas.github.io/Some-Little-JS-Games/异吃棋.html
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import copy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMPTY = 0
X = 1
O = 2
BLOCK = 3

PLAYER_NAMES = {X: 'X', O: 'O'}
CELL_CHARS = {EMPTY: '.', X: 'x', O: 'o', BLOCK: '#'}

# Direct (up/left/right/down) and diagonal (4 corners) neighbor offsets
DIRECT_NEIGHBORS = [(-1, 0), (0, -1), (0, 1), (1, 0)]
DIAGONAL_NEIGHBORS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class GameConfig:
    """Game parameters — see docs/01_rules_formalization.md."""
    board_size: int = 6
    initial_health: int = 2
    attack_power: int = 1
    heal_power: int = 1
    diag_heal: bool = True
    diag_attack: bool = True
    block_density: float = 0.0   # fraction of cells that start as BLOCK

    def __post_init__(self):
        assert self.board_size in (5, 6, 7), f"board_size must be 5/6/7, got {self.board_size}"
        assert self.initial_health >= 1
        assert self.attack_power >= 1
        assert self.heal_power >= 1


# ---------------------------------------------------------------------------
# Game State
# ---------------------------------------------------------------------------
@dataclass
class GameState:
    """Full game state: board types + health + current player + step counter.

    Attributes
    ----------
    config : GameConfig
    types : np.ndarray (N, N) int8   — EMPTY/X/O/BLOCK
    health : np.ndarray (N, N) int8  — 0 for EMPTY/BLOCK, positive otherwise
    current_player : int             — X (1) or O (2)
    step : int                       — number of moves played so far
    """
    config: GameConfig
    types: np.ndarray
    health: np.ndarray
    current_player: int = X
    step: int = 0

    @classmethod
    def initial(cls, config: Optional[GameConfig] = None) -> 'GameState':
        """Create an empty initial state."""
        cfg = config or GameConfig()
        n = cfg.board_size
        s = cls(
            config=cfg,
            types=np.full((n, n), EMPTY, dtype=np.int8),
            health=np.zeros((n, n), dtype=np.int8),
            current_player=X,
            step=0,
        )
        if cfg.block_density > 0:
            s._scatter_blocks()
        return s

    def _scatter_blocks(self):
        """Randomly place BLOCK cells according to block_density (in-place)."""
        rng = np.random.default_rng()
        n = self.config.board_size
        n_blocks = int(round(n * n * self.config.block_density))
        if n_blocks == 0:
            return
        # Pick random cells, ensure first move can still be made (board not all blocks)
        indices = rng.choice(n * n, size=min(n_blocks, n * n - 1), replace=False)
        for idx in indices:
            r, c = divmod(idx, n)
            self.types[r, c] = BLOCK
            self.health[r, c] = 0

    # ----------------------------------------------------------------
    # Cloning & equality
    # ----------------------------------------------------------------
    def clone(self) -> 'GameState':
        return GameState(
            config=self.config,   # immutable, can share
            types=self.types.copy(),
            health=self.health.copy(),
            current_player=self.current_player,
            step=self.step,
        )

    def __eq__(self, other: 'GameState') -> bool:
        return (
            self.config.board_size == other.config.board_size
            and np.array_equal(self.types, other.types)
            and np.array_equal(self.health, other.health)
            and self.current_player == other.current_player
            and self.step == other.step
        )

    def hash(self) -> int:
        """Stable hash for caching."""
        return hash((
            self.config.board_size,
            self.types.tobytes(),
            self.health.tobytes(),
            self.current_player,
            self.step,
        ))

    # ----------------------------------------------------------------
    # Coordinate helpers
    # ----------------------------------------------------------------
    @property
    def board_size(self) -> int:
        return self.config.board_size

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.config.board_size and 0 <= c < self.config.board_size

    def at(self, r: int, c: int) -> int:
        return int(self.types[r, c])

    def hp(self, r: int, c: int) -> int:
        return int(self.health[r, c])

    # ----------------------------------------------------------------
    # Legality & termination
    # ----------------------------------------------------------------
    def legal_moves(self) -> List[Tuple[int, int]]:
        """Return all legal (r, c) moves.

        A cell is legal if:
          - It is EMPTY, AND
          - Either the board has no pieces/blocks yet (opening move),
            OR it lies within 2-Chebyshev distance of some existing piece/block.
        """
        n = self.config.board_size
        # If no pieces/blocks exist, every empty cell is legal (opening).
        non_empty = (self.types != EMPTY)
        if not non_empty.any():
            return [(r, c) for r in range(n) for c in range(n) if self.types[r, c] == EMPTY]

        # Otherwise: collect all cells within 2-Chebyshev distance of any non-empty cell.
        # We use a boolean mask via offset shifts for speed.
        mask = np.zeros((n, n), dtype=bool)
        non_empty_idx = np.argwhere(non_empty)
        for (r, c) in non_empty_idx:
            r_min, r_max = max(0, r - 2), min(n, r + 3)
            c_min, c_max = max(0, c - 2), min(n, c + 3)
            mask[r_min:r_max, c_min:c_max] = True
        mask &= (self.types == EMPTY)
        return [(int(r), int(c)) for r, c in np.argwhere(mask)]

    def is_full(self) -> bool:
        return not (self.types == EMPTY).any()

    def is_terminal(self) -> bool:
        """Terminal if board is full OR no legal moves."""
        if self.is_full():
            return True
        if not self.legal_moves():
            return True
        return False

    def winner(self) -> int:
        """Return X (1), O (2), -1 for draw, or 0 if ongoing."""
        if not self.is_full():
            # Could be terminal due to no legal moves — count pieces
            pass
        else:
            pass
        # Count by pieces
        x_count = int((self.types == X).sum())
        o_count = int((self.types == O).sum())
        if x_count > o_count:
            return X
        elif o_count > x_count:
            return O
        else:
            return -1  # draw

    def terminal_reward_for_current_player(self) -> float:
        """Return +1 if current player wins, -1 if loses, 0 if draw."""
        if not self.is_terminal():
            return 0.0
        w = self.winner()
        if w == -1:
            return 0.0
        return 1.0 if w == self.current_player else -1.0

    # ----------------------------------------------------------------
    # Apply move with chain reaction
    # ----------------------------------------------------------------
    def apply_move(self, move: Tuple[int, int]) -> 'GameState':
        """Apply a move in-place, run chain refresh, switch player.

        Note: this method MUTATES self. Use clone() first if you need to
        preserve the original state.
        """
        r, c = move
        assert self.types[r, c] == EMPTY, f"Cannot place on non-empty cell ({r},{c})"
        p = self.current_player

        # 1. Place stone
        self.types[r, c] = p
        self.health[r, c] = self.config.initial_health

        # 2. Chain refresh until stable
        changed = False
        while True:
            changed = False
            self._refresh_board(changed_flag=[changed])
            if not self._last_changed:
                break

        # 3. Switch player
        self.current_player = O if p == X else X
        self.step += 1
        return self

    # We use a small attribute to communicate "changed" out of _refresh_board.
    # (A cleaner API would return changed, but we want to mirror the JS
    # `boardChanged` global for clarity.)
    _last_changed: bool = False

    def _refresh_board(self, changed_flag):
        """One pass of refresh_board (mirrors JS refreshBoard)."""
        cfg = self.config
        n = cfg.board_size
        # We track "changed" via instance attribute (self._last_changed)
        # because Python list mutation in args is awkward.
        local_changed = False

        # Step A: block_rule (only if asymmetric config)
        if (not cfg.diag_heal) or (not cfg.diag_attack) or (cfg.attack_power != cfg.heal_power):
            self._block_rule()

        # Step B: heal all
        for r in range(n):
            for c in range(n):
                t = self.types[r, c]
                if t != EMPTY and t != BLOCK:
                    self._heal_rule(r, c)

        # Step C: damage + death for current player
        cur = self.current_player
        for r in range(n):
            for c in range(n):
                if self.types[r, c] == cur:
                    self._damage_rule(r, c, cur)
        for r in range(n):
            for c in range(n):
                if self.types[r, c] == cur:
                    if self._death_rule(r, c):
                        local_changed = True

        # Step D: heal all again
        for r in range(n):
            for c in range(n):
                t = self.types[r, c]
                if t != EMPTY and t != BLOCK:
                    self._heal_rule(r, c)

        # Step E: damage + death for opponent
        opp = O if cur == X else X
        for r in range(n):
            for c in range(n):
                if self.types[r, c] == opp:
                    self._damage_rule(r, c, opp)
        for r in range(n):
            for c in range(n):
                if self.types[r, c] == opp:
                    if self._death_rule(r, c):
                        local_changed = True

        self._last_changed = local_changed

    # ----------------------------------------------------------------
    # Individual rules
    # ----------------------------------------------------------------
    def _count_neighbors(self, r: int, c: int, target: int):
        """Count direct/diagonal neighbors of type `target`."""
        n = self.config.board_size
        n_direct = 0
        for dr, dc in DIRECT_NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and self.types[nr, nc] == target:
                n_direct += 1
        m_diag = 0
        for dr, dc in DIAGONAL_NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and self.types[nr, nc] == target:
                m_diag += 1
        return n_direct, m_diag

    def _heal_rule(self, r: int, c: int):
        """Reset health; if ≥2 same-color direct neighbors, add bonus."""
        cfg = self.config
        me = self.types[r, c]
        if me == EMPTY or me == BLOCK:
            return
        # Reset to base health
        self.health[r, c] = cfg.initial_health
        n, m = self._count_neighbors(r, c, me)
        if n >= 2:
            if m > 0 and cfg.diag_heal:
                self.health[r, c] = cfg.initial_health + (n + m) * cfg.heal_power - 1
            else:
                self.health[r, c] = cfg.initial_health + n * cfg.heal_power - 1

    def _damage_rule(self, r: int, c: int, player: int):
        """If ≥2 opposite-color direct neighbors, reduce health."""
        cfg = self.config
        opp = O if player == X else X
        # The piece at (r,c) is `player`; count its opponent neighbors
        n, m = self._count_neighbors(r, c, opp)
        if n >= 2:
            if m > 0 and cfg.diag_attack:
                self.health[r, c] -= (n + m) * cfg.attack_power
            else:
                self.health[r, c] -= n * cfg.attack_power

    def _death_rule(self, r: int, c: int) -> bool:
        """If health ≤ 0, flip to opponent color, reset health to 2."""
        if self.types[r, c] == EMPTY or self.types[r, c] == BLOCK:
            return False
        if self.health[r, c] <= 0:
            self.types[r, c] = O if self.types[r, c] == X else X
            # Mirror JS: health = 2 (NOT inithealth, see game_source.js line 730)
            self.health[r, c] = 2
            return True
        return False

    def _block_rule(self):
        """Cells with ≥2 X and ≥2 O direct neighbors become BLOCK."""
        cfg = self.config
        n = cfg.board_size
        for r in range(n):
            for c in range(n):
                if self.types[r, c] == BLOCK:
                    continue
                x_cnt = 0
                o_cnt = 0
                for dr, dc in DIRECT_NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        if self.types[nr, nc] == X:
                            x_cnt += 1
                        elif self.types[nr, nc] == O:
                            o_cnt += 1
                if x_cnt >= 2 and o_cnt >= 2:
                    self.types[r, c] = BLOCK
                    self.health[r, c] = 0

    # ----------------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------------
    def to_string(self) -> str:
        n = self.config.board_size
        lines = []
        # Header
        header = "   " + " ".join(chr(ord('A') + i) for i in range(n))
        lines.append(header)
        for r in range(n):
            row_label = f"{r+1:>2} "
            cells = []
            for c in range(n):
                t = self.types[r, c]
                ch = CELL_CHARS[int(t)]
                if t in (X, O):
                    h = self.health[r, c]
                    cells.append(f"{ch}{h}")
                else:
                    cells.append(f"{ch} ")
            lines.append(row_label + " ".join(cells))
        lines.append(f"Current: {PLAYER_NAMES[self.current_player]}, step: {self.step}")
        return "\n".join(lines)

    def __str__(self):
        return self.to_string()


# ---------------------------------------------------------------------------
# Feature planes (network input encoding)
# ---------------------------------------------------------------------------
# 11 channels (perspective-aligned to current player):
#   0: my_stones
#   1: opp_stones
#   2: blocks
#   3: my_health_1   (health == 1)
#   4: my_health_2   (health == 2)
#   5: my_health_3p  (health >= 3)
#   6: opp_health_1
#   7: opp_health_2
#   8: opp_health_3p
#   9: legal_moves
#  10: side_to_move  (constant 1 after alignment)
N_FEATURE_CHANNELS = 11


def state_to_tensor(state: GameState) -> np.ndarray:
    """Convert state to (C, N, N) float32 tensor, perspective-aligned."""
    cfg = state.config
    n = cfg.board_size
    p = state.current_player
    opp = O if p == X else X

    planes = np.zeros((N_FEATURE_CHANNELS, n, n), dtype=np.float32)

    for r in range(n):
        for c in range(n):
            t = state.types[r, c]
            h = state.health[r, c]
            if t == p:
                planes[0, r, c] = 1
                _fill_health_bucket(planes, 3, 4, 5, h)
            elif t == opp:
                planes[1, r, c] = 1
                _fill_health_bucket(planes, 6, 7, 8, h)
            elif t == BLOCK:
                planes[2, r, c] = 1

    # Legal moves
    for (r, c) in state.legal_moves():
        planes[9, r, c] = 1

    # Side to move (constant 1 after perspective alignment)
    planes[10, :, :] = 1

    return planes


def _fill_health_bucket(planes, idx_h1, idx_h2, idx_h3p, h: int):
    if h == 1:
        planes[idx_h1] = 1  # caller assigns per-cell, so this is wrong;
        # actually we need to set the (r, c) cell, but here we don't have r, c.
        # Fix: caller should pass r, c too. Refactor below.
        pass
    # This helper is unused; actual filling is done inline below.


def state_to_tensor_v2(state: GameState) -> np.ndarray:
    """Cleaner implementation of state_to_tensor."""
    cfg = state.config
    n = cfg.board_size
    p = state.current_player
    opp = O if p == X else X

    planes = np.zeros((N_FEATURE_CHANNELS, n, n), dtype=np.float32)

    for r in range(n):
        for c in range(n):
            t = state.types[r, c]
            h = int(state.health[r, c])
            if t == p:
                planes[0, r, c] = 1
                if h == 1:        planes[3, r, c] = 1
                elif h == 2:      planes[4, r, c] = 1
                elif h >= 3:      planes[5, r, c] = 1
            elif t == opp:
                planes[1, r, c] = 1
                if h == 1:        planes[6, r, c] = 1
                elif h == 2:      planes[7, r, c] = 1
                elif h >= 3:      planes[8, r, c] = 1
            elif t == BLOCK:
                planes[2, r, c] = 1

    for (r, c) in state.legal_moves():
        planes[9, r, c] = 1

    planes[10, :, :] = 1
    return planes


# Replace state_to_tensor with the cleaner version
state_to_tensor = state_to_tensor_v2


def move_to_index(move: Tuple[int, int], board_size: int) -> int:
    """Convert (r, c) to flat index in [0, N*N). Pass = N*N."""
    return move[0] * board_size + move[1]


def index_to_move(idx: int, board_size: int) -> Tuple[int, int]:
    return (idx // board_size, idx % board_size)


# ---------------------------------------------------------------------------
# Symmetry (D4 group: 4 rotations × 2 flips = 8 transforms)
# ---------------------------------------------------------------------------
def apply_symmetry_to_planes(planes: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """Apply k rotations (counter-clockwise) and optional horizontal flip to (C, N, N)."""
    out = np.rot90(planes, k, axes=(1, 2))
    if flip:
        out = np.flip(out, axis=2)
    return out.copy()


def apply_symmetry_to_policy(policy: np.ndarray, board_size: int,
                              k: int, flip: bool) -> np.ndarray:
    """Apply same symmetry to policy vector (length N*N+1).

    The pass action (last index) is invariant under symmetry.
    """
    n_actions = board_size * board_size + 1
    assert len(policy) == n_actions, f"policy length {len(policy)} != {n_actions}"
    # Reshape first N*N entries into (N, N), keep pass separate
    grid = policy[:-1].reshape(board_size, board_size)
    pass_p = policy[-1]
    grid = np.rot90(grid, k)
    if flip:
        grid = np.flip(grid, axis=1)
    out = np.empty(n_actions, dtype=policy.dtype)
    out[:-1] = grid.reshape(-1)
    out[-1] = pass_p
    return out


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Sanity check: empty board, first move at (2, 2) should be legal everywhere
    s = GameState.initial()
    print("Initial state:")
    print(s)
    print()
    print(f"Legal moves count: {len(s.legal_moves())} (expect {6*6})")
    print()

    # Apply a move at center
    s.apply_move((2, 2))
    print("After X plays (2,2):")
    print(s)
    print(f"Legal moves count: {len(s.legal_moves())}")
    print()

    # Apply a move at (2, 3) by O
    s.apply_move((2, 3))
    print("After O plays (2,3):")
    print(s)
    print()

    # Test feature tensor
    planes = state_to_tensor(s)
    print(f"Feature tensor shape: {planes.shape}, dtype: {planes.dtype}")
    print(f"Non-zero channels: {[(i, int(planes[i].sum())) for i in range(11) if planes[i].sum() > 0]}")
