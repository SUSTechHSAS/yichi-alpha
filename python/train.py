"""
YichiAlpha — Training Loop
============================
Self-play → Replay buffer → SGD training → Checkpoint → Evaluation.

Usage:
    python3 train.py --config ../configs/default.yaml --iterations 5

Config keys (see configs/default.yaml):
    board_size, initial_health, ...
    in_channels, channels, n_blocks
    n_simulations_train, n_simulations_eval
    c_puct, dirichlet_alpha, dirichlet_epsilon
    iterations, selfplay_games_per_iter, train_batches_per_iter
    batch_size, lr, weight_decay, lr_min
    buffer_capacity
    eval_every, arena_games, win_rate_threshold
    checkpoint_dir, log_dir
    cold_start_random_games   # bootstrap buffer with random games
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import json
import math
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import multiprocessing as mp

# Add this file's directory to path
sys.path.insert(0, str(Path(__file__).parent))

from game import GameState, GameConfig, X, O
from model import YichiNet, save_checkpoint, load_checkpoint
from mcts import MCTS
from dataset import ReplayBuffer
from selfplay import self_play_game, self_play_random_baseline
from evaluate import MCTSAgent, RandomAgent, play_match
from load_cpp_data import load_cpp_selfplay_dir


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # Game
    board_size: int = 6
    initial_health: int = 2
    attack_power: int = 1
    heal_power: int = 1
    diag_heal: bool = True
    diag_attack: bool = True

    # Network
    in_channels: int = 11
    channels: int = 64
    n_blocks: int = 6

    # MCTS
    n_simulations_train: int = 100
    n_simulations_eval: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # Training
    iterations: int = 5
    selfplay_games_per_iter: int = 4
    train_batches_per_iter: int = 20
    batch_size: int = 64
    lr: float = 0.01
    weight_decay: float = 1e-4
    lr_min: float = 1e-4
    grad_clip: float = 1.0

    # Buffer
    buffer_capacity: int = 10000
    cold_start_random_games: int = 10  # bootstrap with random games

    # Eval
    eval_every: int = 5
    arena_games: int = 10
    win_rate_threshold: float = 0.55

    # IO
    checkpoint_dir: str = "../checkpoints"
    log_dir: str = "../logs"
    device: str = "cpu"
    seed: int = 42

    # C++ data loading (optional — if set, skip Python self-play and train from C++ data)
    # Set this to a directory containing game_*.bin files produced by yichi_selfplay.
    # The training loop will load ALL .bin files at startup, then train without self-play.
    # Use this for distributed training: C++ engine generates data, Python trains.
    cpp_data_dir: str = ""       # empty = use Python self-play (default)
    cpp_data_refresh: bool = False  # if True, re-scan cpp_data_dir every iteration

    # Async evaluation (recommended for CPU — speeds up training 3-5x)
    # When True, eval and Arena run in a forked subprocess that doesn't block training.
    # Results are written to eval_result_iter{N}.json and read back next iteration.
    async_eval: bool = True
    eval_n_games: int = 8        # games for vs-random eval (per iter)

    @classmethod
    def from_yaml(cls, path: str) -> 'TrainConfig':
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def game_config(self) -> GameConfig:
        return GameConfig(
            board_size=self.board_size,
            initial_health=self.initial_health,
            attack_power=self.attack_power,
            heal_power=self.heal_power,
            diag_heal=self.diag_heal,
            diag_attack=self.diag_attack,
        )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def compute_loss(model: YichiNet, states, target_policies, target_values,
                  l2_coef: float = 1e-4):
    """Standard AlphaZero loss: policy CE + value MSE + L2."""
    policy_logits, value = model(states)

    # Policy: cross-entropy with soft targets
    log_policy = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -(target_policies * log_policy).sum(dim=-1).mean()

    # Value: MSE
    value_loss = F.mse_loss(value, target_values)

    # L2 (manual, since we use weight_decay in optimizer too — but explicit helps debugging)
    l2 = sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)
    l2_loss = l2_coef * l2

    total = policy_loss + value_loss + l2_loss
    return total, policy_loss.item(), value_loss.item()


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
class Logger:
    def __init__(self, log_path: Optional[Path] = None, jsonl_path: Optional[Path] = None):
        self.log_path = log_path
        self.jsonl_path = jsonl_path
        self.history = []
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = open(log_path, 'a')
        else:
            self.log_file = None
        if jsonl_path:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_file = open(jsonl_path, 'a')
        else:
            self.jsonl_file = None

    def log(self, msg: str, extra: Optional[Dict] = None):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        if self.log_file:
            self.log_file.write(line + "\n")
            self.log_file.flush()
        if extra and self.jsonl_file:
            entry = {"ts": ts, "msg": msg, **extra}
            self.jsonl_file.write(json.dumps(entry) + "\n")
            self.jsonl_file.flush()
            self.history.append(entry)

    def close(self):
        if self.log_file: self.log_file.close()
        if self.jsonl_file: self.jsonl_file.close()


# ---------------------------------------------------------------------------
# Async evaluation worker (runs in a forked subprocess)
# ---------------------------------------------------------------------------
def _eval_worker(model_path: str, best_model_path: str, config_dict: dict,
                  result_path: str, iteration: int, do_arena: bool):
    """Subprocess: load model from file, run eval + optional Arena, write result JSON.

    This runs in a forked process so it doesn't block training.
    Reads model from disk (no shared memory with trainer).
    """
    # Rebuild config from dict
    cfg = TrainConfig(**config_dict)
    game_cfg = GameConfig(
        board_size=cfg.board_size, initial_health=cfg.initial_health,
        attack_power=cfg.attack_power, heal_power=cfg.heal_power,
        diag_heal=cfg.diag_heal, diag_attack=cfg.diag_attack,
    )

    result = {"iter": iteration, "vs_random": None, "arena": None}

    try:
        model = load_checkpoint(model_path, device=cfg.device)
        model_agent = MCTSAgent(
            model, n_simulations=cfg.n_simulations_eval,
            temperature=0.0, device=cfg.device,
        )
        rng = np.random.default_rng(cfg.seed + 9999 + iteration)
        random_agent = RandomAgent(rng=rng)

        # vs random
        n_games = cfg.eval_n_games
        wins = losses_n = draws = 0
        for g in range(n_games):
            if g % 2 == 0:
                w = play_match(model_agent, random_agent, game_cfg)
                if w == X: wins += 1
                elif w == O: losses_n += 1
                else: draws += 1
            else:
                w = play_match(random_agent, model_agent, game_cfg)
                if w == O: wins += 1
                elif w == X: losses_n += 1
                else: draws += 1
        result["vs_random"] = {
            "wins": wins, "losses": losses_n, "draws": draws,
            "n_games": n_games, "win_rate": wins / n_games,
        }

        # Arena: new vs best
        if do_arena and Path(best_model_path).exists():
            best_model = load_checkpoint(best_model_path, device=cfg.device)
            best_agent = MCTSAgent(
                best_model, n_simulations=cfg.n_simulations_eval,
                temperature=0.0, device=cfg.device,
            )
            a_wins = b_wins = a_draws = 0
            for g in range(cfg.arena_games):
                if g % 2 == 0:
                    w = play_match(model_agent, best_agent, game_cfg)
                    if w == X: a_wins += 1
                    elif w == O: b_wins += 1
                    else: a_draws += 1
                else:
                    w = play_match(best_agent, model_agent, game_cfg)
                    if w == O: a_wins += 1
                    elif w == X: b_wins += 1
                    else: a_draws += 1
            new_wr = a_wins / cfg.arena_games
            result["arena"] = {
                "new_wins": a_wins, "best_wins": b_wins, "draws": a_draws,
                "n_games": cfg.arena_games, "new_win_rate": new_wr,
                "new_is_better": new_wr > cfg.win_rate_threshold,
            }
    except Exception as e:
        result["error"] = str(e)

    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)


def _check_pending_evals(eval_dir: Path, logger: Logger):
    """Check for completed async eval result files and log them."""
    for result_file in sorted(eval_dir.glob("eval_result_iter*.json")):
        try:
            with open(result_file) as f:
                result = json.load(f)
            iter_num = result["iter"]
            # Check if we already logged this one (file has .logged marker)
            marker = result_file.with_suffix('.logged')
            if marker.exists():
                continue

            if result.get("vs_random"):
                vr = result["vs_random"]
                logger.log(
                    f"  [async eval] Iter {iter_num} vs random: "
                    f"{vr['wins']}/{vr['n_games']} ({vr['win_rate']:.1%}), "
                    f"{vr['losses']} losses, {vr['draws']} draws",
                    extra={"iter": iter_num, "phase": "async_eval",
                           **{f"vs_random_{k}": v for k, v in vr.items()}}
                )
            if result.get("arena"):
                ar = result["arena"]
                logger.log(
                    f"  [async arena] Iter {iter_num}: new vs best "
                    f"{ar['new_wins']}/{ar['n_games']} ({ar['new_win_rate']:.1%})",
                    extra={"iter": iter_num, "phase": "async_arena",
                           **{f"arena_{k}": v for k, v in ar.items()}}
                )
            marker.touch()
            # Optionally delete the result file
            # result_file.unlink()
        except Exception as e:
            logger.log(f"  [async eval] failed to read {result_file}: {e}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(config: TrainConfig, logger: Optional[Logger] = None):
    """Main training loop."""
    if logger is None:
        logger = Logger()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device(config.device)
    logger.log(f"Training config: {asdict(config)}")

    # Game config
    game_cfg = config.game_config()

    # Model
    model = YichiNet(
        board_size=config.board_size,
        in_channels=config.in_channels,
        channels=config.channels,
        n_blocks=config.n_blocks,
    ).to(device)
    logger.log(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # MCTS
    mcts = MCTS(
        model, c_puct=config.c_puct,
        n_simulations=config.n_simulations_train,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
        device=config.device,
    )

    # Buffer
    buffer = ReplayBuffer(capacity=config.buffer_capacity, board_size=config.board_size)

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.lr,
        momentum=0.9, weight_decay=config.weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.iterations, eta_min=config.lr_min,
    )

    # Checkpoint dir
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Eval dir (for async eval result files)
    eval_dir = Path(config.log_dir) / "eval_results"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Use 'spawn' for multiprocessing to avoid issues with CUDA/fork
    if config.async_eval:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # already set

    # ----- Cold start: bootstrap buffer with random games -----
    if config.cold_start_random_games > 0:
        logger.log(f"Cold start: generating {config.cold_start_random_games} random games...")
        rng = np.random.default_rng(config.seed)
        total_samples = 0
        for g in range(config.cold_start_random_games):
            samples = self_play_random_baseline(game_cfg, rng=rng)
            buffer.push(samples)
            total_samples += len(samples)
        logger.log(f"Cold start done: {total_samples} samples in buffer")

    # Save initial model
    save_checkpoint(model, str(ckpt_dir / "model_iter0.pt"),
                    extra={"iter": 0, "config": asdict(config)})
    save_checkpoint(model, str(ckpt_dir / "model_best.pt"),
                    extra={"iter": 0, "config": asdict(config)})
    logger.log(f"Saved initial model to {ckpt_dir / 'model_iter0.pt'}")

    # ----- Optional: preload C++ self-play data -----
    use_cpp_data = bool(config.cpp_data_dir)
    if use_cpp_data:
        logger.log(f"Loading C++ self-play data from {config.cpp_data_dir}...")
        cpp_samples = load_cpp_selfplay_dir(config.cpp_data_dir, verbose=True)
        buffer.push(cpp_samples)
        logger.log(f"C++ data loaded: {len(cpp_samples)} samples, buffer size = {len(buffer)}")
        if len(cpp_samples) == 0:
            logger.log("WARNING: no C++ data found, falling back to Python self-play")
            use_cpp_data = False

    # ----- Main loop -----
    for iteration in range(1, config.iterations + 1):
        iter_t0 = time.time()

        # --- 1. Self-play (skip if using C++ data) ---
        if use_cpp_data:
            if config.cpp_data_refresh:
                # Re-scan directory for new games (for ongoing C++ generation)
                new_samples = load_cpp_selfplay_dir(
                    config.cpp_data_dir, verbose=False
                )
                buffer.push(new_samples)
                logger.log(
                    f"Iter {iteration}: refreshed {len(new_samples)} C++ samples, "
                    f"buffer size = {len(buffer)}",
                    extra={"iter": iteration, "phase": "cpp_data_load",
                           "n_samples": len(new_samples), "buffer_size": len(buffer)}
                )
            else:
                logger.log(
                    f"Iter {iteration}: using preloaded C++ data (buffer size = {len(buffer)})",
                    extra={"iter": iteration, "phase": "cpp_data_skip",
                           "buffer_size": len(buffer)}
                )
        else:
            # Python self-play (original path)
            model.eval()
            rng = np.random.default_rng(config.seed + iteration)
            all_samples = []
            sp_t0 = time.time()
            for g in range(config.selfplay_games_per_iter):
                samples = self_play_game(model, mcts, game_cfg, rng=rng)
                all_samples.extend(samples)
            sp_dt = time.time() - sp_t0
            buffer.push(all_samples)
            logger.log(
                f"Iter {iteration}: self-play generated {len(all_samples)} samples "
                f"({config.selfplay_games_per_iter} games, {sp_dt:.1f}s), "
                f"buffer size = {len(buffer)}",
                extra={"iter": iteration, "phase": "selfplay",
                       "n_samples": len(all_samples), "buffer_size": len(buffer),
                       "selfplay_time": sp_dt}
            )

        # --- 2. Training ---
        model.train()
        train_t0 = time.time()
        losses = []
        policy_losses = []
        value_losses = []
        for step in range(config.train_batches_per_iter):
            states, target_pi, target_v = buffer.sample_with_augmentation(config.batch_size)
            states = states.to(device)
            target_pi = target_pi.to(device)
            target_v = target_v.to(device)

            loss, p_loss, v_loss = compute_loss(
                model, states, target_pi, target_v,
                l2_coef=config.weight_decay,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()

            losses.append(loss.item())
            policy_losses.append(p_loss)
            value_losses.append(v_loss)
        scheduler.step()
        train_dt = time.time() - train_t0
        avg_loss = float(np.mean(losses))
        avg_p = float(np.mean(policy_losses))
        avg_v = float(np.mean(value_losses))
        logger.log(
            f"Iter {iteration}: training done ({train_dt:.1f}s) "
            f"loss={avg_loss:.4f} (p={avg_p:.4f}, v={avg_v:.4f}) "
            f"lr={scheduler.get_last_lr()[0]:.5f}",
            extra={"iter": iteration, "phase": "train",
                   "loss": avg_loss, "policy_loss": avg_p, "value_loss": avg_v,
                   "lr": scheduler.get_last_lr()[0], "train_time": train_dt}
        )

        # --- 3. Checkpoint ---
        ckpt_path = ckpt_dir / f"model_iter{iteration}.pt"
        save_checkpoint(model, str(ckpt_path),
                        extra={"iter": iteration, "config": asdict(config)})
        logger.log(f"Iter {iteration}: saved checkpoint to {ckpt_path}")

        # --- 4. Evaluation (async or sync) ---
        # Check for any completed async evals from previous iterations
        if config.async_eval:
            _check_pending_evals(eval_dir, logger)

        do_arena = (iteration % config.eval_every == 0 and iteration > 0)

        if config.async_eval:
            # Fork a subprocess to run eval + arena; don't block training
            result_path = eval_dir / f"eval_result_iter{iteration}.json"
            # Clean up any stale marker
            marker = result_path.with_suffix('.logged')
            if marker.exists():
                marker.unlink()
            if result_path.exists():
                result_path.unlink()

            best_path_str = str(ckpt_dir / "model_best.pt")
            config_dict = asdict(config)

            proc = mp.Process(
                target=_eval_worker,
                args=(str(ckpt_path), best_path_str, config_dict,
                      str(result_path), iteration, do_arena),
                daemon=True,
            )
            proc.start()
            logger.log(
                f"Iter {iteration}: async eval forked (PID {proc.pid}), "
                f"result will appear as [async eval] when done",
                extra={"iter": iteration, "phase": "async_eval_fork",
                       "pid": proc.pid}
            )
            # Don't join — let it run in background
        else:
            # Synchronous eval (original behavior, blocks training)
            eval_t0 = time.time()
            n_eval_games = config.eval_n_games
            wins = losses_n = draws = 0
            rng_eval = np.random.default_rng(config.seed + 9999 + iteration)
            random_agent = RandomAgent(rng=rng_eval)
            model_agent = MCTSAgent(
                model, n_simulations=config.n_simulations_eval,
                temperature=0.0, device=config.device,
            )
            for g in range(n_eval_games):
                if g % 2 == 0:
                    w = play_match(model_agent, random_agent, game_cfg)
                    if w == X: wins += 1
                    elif w == O: losses_n += 1
                    else: draws += 1
                else:
                    w = play_match(random_agent, model_agent, game_cfg)
                    if w == O: wins += 1
                    elif w == X: losses_n += 1
                    else: draws += 1
            eval_dt = time.time() - eval_t0
            win_rate = wins / n_eval_games
            logger.log(
                f"Iter {iteration}: eval vs random: {wins}/{n_eval_games} wins "
                f"({win_rate:.1%}), {losses_n} losses, {draws} draws ({eval_dt:.1f}s)",
                extra={"iter": iteration, "phase": "eval",
                       "win_rate": win_rate, "wins": wins, "losses": losses_n,
                       "draws": draws, "eval_time": eval_dt}
            )

            # Arena (sync)
            if do_arena:
                arena_t0 = time.time()
                best_path = ckpt_dir / "model_best.pt"
                if best_path.exists():
                    best_model = load_checkpoint(str(best_path), device=config.device)
                    best_agent = MCTSAgent(
                        best_model, n_simulations=config.n_simulations_eval,
                        temperature=0.0, device=config.device,
                    )
                    a_wins = b_wins = a_draws = 0
                    for g in range(config.arena_games):
                        if g % 2 == 0:
                            w = play_match(model_agent, best_agent, game_cfg)
                            if w == X: a_wins += 1
                            elif w == O: b_wins += 1
                            else: a_draws += 1
                        else:
                            w = play_match(best_agent, model_agent, game_cfg)
                            if w == O: a_wins += 1
                            elif w == X: b_wins += 1
                            else: a_draws += 1
                    arena_dt = time.time() - arena_t0
                    new_win_rate = a_wins / config.arena_games
                    logger.log(
                        f"Iter {iteration}: ARENA new vs best: {a_wins}/{config.arena_games} "
                        f"({new_win_rate:.1%}), {b_wins} losses, {a_draws} draws ({arena_dt:.1f}s)",
                        extra={"iter": iteration, "phase": "arena",
                               "new_win_rate": new_win_rate, "a_wins": a_wins,
                               "b_wins": b_wins, "draws": a_draws, "arena_time": arena_dt}
                    )
                    if new_win_rate > config.win_rate_threshold:
                        save_checkpoint(model, str(best_path),
                                        extra={"iter": iteration, "config": asdict(config)})
                        logger.log(f"Iter {iteration}: NEW BEST MODEL saved")
                else:
                    save_checkpoint(model, str(best_path),
                                    extra={"iter": iteration, "config": asdict(config)})

        iter_dt = time.time() - iter_t0
        logger.log(f"Iter {iteration}: total time {iter_dt:.1f}s")

    # --- After loop: wait for any remaining async evals and log them ---
    if config.async_eval:
        logger.log("Waiting for remaining async evals to finish...")
        # Give them up to 5 minutes
        deadline = time.time() + 300
        while time.time() < deadline:
            _check_pending_evals(eval_dir, logger)
            # Check if any unlogged results remain
            pending = [f for f in eval_dir.glob("eval_result_iter*.json")
                       if not f.with_suffix('.logged').exists()]
            if not pending:
                break
            time.sleep(5)
        _check_pending_evals(eval_dir, logger)

        # Promote best model based on async arena results
        _promote_best_from_async(eval_dir, ckpt_dir, config, logger)

    # Save final
    save_checkpoint(model, str(ckpt_dir / "model_final.pt"),
                    extra={"iter": config.iterations, "config": asdict(config)})
    logger.log(f"Training complete. Final model saved to {ckpt_dir / 'model_final.pt'}")

    return model


def _promote_best_from_async(eval_dir: Path, ckpt_dir: Path,
                              config: TrainConfig, logger: Logger):
    """Scan async arena results and promote the best model to model_best.pt."""
    best_iter = 0
    best_win_rate = config.win_rate_threshold
    for result_file in sorted(eval_dir.glob("eval_result_iter*.json")):
        try:
            with open(result_file) as f:
                result = json.load(f)
            if result.get("arena") and result["arena"]["new_is_better"]:
                if result["arena"]["new_win_rate"] > best_win_rate:
                    best_win_rate = result["arena"]["new_win_rate"]
                    best_iter = result["iter"]
        except Exception:
            pass
    if best_iter > 0:
        src = ckpt_dir / f"model_iter{best_iter}.pt"
        dst = ckpt_dir / "model_best.pt"
        if src.exists():
            import shutil
            shutil.copy2(str(src), str(dst))
            logger.log(f"Promoted model_iter{best_iter}.pt to model_best.pt "
                       f"(arena win rate {best_win_rate:.1%})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="YichiAlpha training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard training (Python self-play + train)
  python train.py --config ../configs/quick.yaml --iterations 5

  # Train from C++ self-play data (no Python self-play — much faster)
  # First generate data with: ./yichi_selfplay --model model.pt --output ./data/
  python train.py --config ../configs/quick.yaml --data_dir ./data/ --iterations 20

  # Train on GPU
  python train.py --config ../configs/default.yaml --device cuda --iterations 100
        """
    )
    parser.add_argument('--config', type=str, default='../configs/default.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Override iterations count')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device (cpu/cuda)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory of C++ self-play .bin files. If set, '
                             'skip Python self-play and train only from this data.')
    parser.add_argument('--data_refresh', action='store_true',
                        help='Re-scan --data_dir every iteration (for ongoing C++ generation)')
    parser.add_argument('--sync_eval', action='store_true',
                        help='Disable async eval (run eval synchronously, blocks training)')
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    if args.iterations is not None:
        config.iterations = args.iterations
    if args.device is not None:
        config.device = args.device
    if args.data_dir is not None:
        config.cpp_data_dir = args.data_dir
    if args.data_refresh:
        config.cpp_data_refresh = True
    if args.sync_eval:
        config.async_eval = False

    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(
        log_path=log_dir / "train.log",
        jsonl_path=log_dir / "train_metrics.jsonl",
    )
    train(config, logger=logger)
    logger.close()


if __name__ == '__main__':
    main()
