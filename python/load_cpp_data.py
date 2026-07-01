"""
load_cpp_data.py — 把 C++ 引擎生成的 .bin 自对弈数据加载回 Python
====================================================================

C++ selfplay_main.cpp 把每局棋保存成 game_*.bin，格式如下：
  int   n_samples
  int   winner
  for each sample:
    int   state_bytes_size
    char  state_bytes[state_bytes_size]   (N, current_player, step, N*N types, N*N health)
    int   policy_size                      (N*N+1)
    float policy[policy_size]
    float value
    int   player

本模块把这个格式读回 Python 的 (GameState, policy, value) 元组，
可以直接 push 到 ReplayBuffer。

用法:
    from load_cpp_data import load_cpp_selfplay_dir
    samples = load_cpp_selfplay_dir('./selfplay_data/')
    buffer.push(samples)
"""
from __future__ import annotations

import struct
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))

from game import GameState, GameConfig, X, O, BLOCK, EMPTY
from dataset import RawSample


# ---------------------------------------------------------------------------
# Binary format reader
# ---------------------------------------------------------------------------
def _read_int(f) -> int:
    """Read a 4-byte little-endian int."""
    data = f.read(4)
    if len(data) < 4:
        raise EOFError("Unexpected end of file reading int")
    return struct.unpack('<i', data)[0]


def _read_float(f) -> float:
    """Read a 4-byte little-endian float."""
    data = f.read(4)
    if len(data) < 4:
        raise EOFError("Unexpected end of file reading float")
    return struct.unpack('<f', data)[0]


def _read_bytes(f, n: int) -> bytes:
    """Read exactly n bytes."""
    data = f.read(n)
    if len(data) < n:
        raise EOFError(f"Unexpected end of file reading {n} bytes")
    return data


def _parse_state_bytes(state_bytes: bytes) -> GameState:
    """Parse the state byte format written by C++ board_to_bytes().

    Format: N (1 byte), current_player (1 byte), step (1 byte),
            N*N types (1 byte each), N*N health (1 byte each)
    """
    if len(state_bytes) < 3:
        raise ValueError(f"State bytes too short: {len(state_bytes)}")

    N = state_bytes[0]
    current_player_raw = state_bytes[1]
    step = state_bytes[2]

    # Map C++ CellType enum to Python constants
    # C++: EMPTY=0, X=1, O=2, BLOCK=3
    cpp_to_py = {0: EMPTY, 1: X, 2: O, 3: BLOCK}

    # current_player in C++ is 1 (X) or 2 (O)
    current_player = X if current_player_raw == 1 else O

    expected_size = 3 + 2 * N * N
    if len(state_bytes) != expected_size:
        raise ValueError(
            f"State bytes size mismatch: got {len(state_bytes)}, expected {expected_size} "
            f"(N={N})"
        )

    cfg = GameConfig(board_size=N)
    state = GameState.initial(cfg)
    state.current_player = current_player
    state.step = step

    offset = 3
    for i in range(N * N):
        raw_type = state_bytes[offset + i]
        state.types.flat[i] = cpp_to_py.get(raw_type, EMPTY)
    offset += N * N
    for i in range(N * N):
        # health is signed in C++ (int8_t), but we stored as char.
        # Values 0-127 are fine; if it was negative it'd be wrong, but health >= 0 always.
        raw_h = state_bytes[offset + i]
        state.health.flat[i] = raw_h if raw_h < 128 else raw_h - 256

    return state


def load_cpp_game_file(path: str) -> List[RawSample]:
    """Load one .bin file produced by C++ selfplay_main.

    Returns a list of (GameState, policy_np, value_float) samples.
    """
    samples = []
    with open(path, 'rb') as f:
        n_samples = _read_int(f)
        winner = _read_int(f)  # noqa: F841 (we use values directly, not winner)

        for _ in range(n_samples):
            # State bytes
            sb_size = _read_int(f)
            state_bytes = _read_bytes(f, sb_size)
            state = _parse_state_bytes(state_bytes)

            # Policy
            p_size = _read_int(f)
            policy_bytes = _read_bytes(f, p_size * 4)  # float = 4 bytes
            policy = np.frombuffer(policy_bytes, dtype='<f4').astype(np.float32)
            assert len(policy) == p_size, f"Policy size mismatch: {len(policy)} vs {p_size}"

            # Value
            value = _read_float(f)

            # Player (we don't actually need this — value is already perspective-aligned)
            _player = _read_int(f)  # noqa: F841

            samples.append((state, policy, value))

    return samples


def load_cpp_selfplay_dir(dir_path: str,
                           pattern: str = 'game_*.bin',
                           max_files: Optional[int] = None,
                           verbose: bool = True) -> List[RawSample]:
    """Load all .bin files from a directory.

    Parameters
    ----------
    dir_path : str
        Directory containing game_*.bin files
    pattern : str
        Glob pattern for game files
    max_files : int, optional
        If set, only load the first N files (useful for testing)
    verbose : bool
        Print progress

    Returns
    -------
    List of (GameState, policy, value) samples, ready for buffer.push()
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    files = sorted(dir_path.glob(pattern))
    if max_files is not None:
        files = files[:max_files]

    if verbose:
        print(f"Loading {len(files)} game files from {dir_path}...")

    all_samples = []
    for i, f in enumerate(files):
        try:
            samples = load_cpp_game_file(str(f))
            all_samples.extend(samples)
        except Exception as e:
            if verbose:
                print(f"  WARNING: failed to load {f.name}: {e}")
            continue
        if verbose and (i + 1) % 50 == 0:
            print(f"  Loaded {i+1}/{len(files)} files, {len(all_samples)} samples so far")

    if verbose:
        print(f"Total: {len(all_samples)} samples from {len(files)} files")
    return all_samples


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python load_cpp_data.py <selfplay_dir> [max_files]")
        print("Example: python load_cpp_data.py /tmp/yichi_test_output 5")
        sys.exit(1)

    sp_dir = sys.argv[1]
    max_files = int(sys.argv[2]) if len(sys.argv) > 2 else None

    samples = load_cpp_selfplay_dir(sp_dir, max_files=max_files)
    if not samples:
        print("No samples loaded. Make sure the directory contains game_*.bin files")
        print("generated by: ./yichi_selfplay --model ... --output <dir>")
        sys.exit(1)

    print(f"\nFirst sample:")
    s, p, v = samples[0]
    print(f"  State:\n{s}")
    print(f"  Policy shape: {p.shape}, sum: {p.sum():.4f}")
    print(f"  Policy top-3: {np.argsort(p)[::-1][:3]}")
    print(f"  Value: {v:+.3f}")

    print(f"\nLast sample:")
    s, p, v = samples[-1]
    print(f"  State step: {s.step}")
    print(f"  Value: {v:+.3f}  (winner's perspective)")

    print(f"\nTotal samples loaded: {len(samples)}")
    print("✓ C++ data can be loaded into Python ReplayBuffer")
