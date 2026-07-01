# 05 · 训练 Pipeline (Training Pipeline)

> 自对弈 → 数据缓冲 → 训练 → 评估 → 检查点 的完整闭环。
>
> 借鉴 KataGo 的 "loop until convergence" 训练范式，但简化为单机 CPU 版本。

## 1. 总体流程

```
┌────────────────────────────────────────────────────────────────┐
│                       训练主循环                                │
│                                                                │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│   │  Self-Play   │───▶│  Data Buffer │───▶│   Training   │    │
│   │  (最新模型)  │    │  (FIFO 50k)  │    │   (SGD step) │    │
│   └──────────────┘    └──────────────┘    └──────────────┘    │
│         ▲                                          │           │
│         │                                          ▼           │
│   ┌──────────────┐                          ┌──────────────┐   │
│   │  Checkpoint  │◀─────────────────────────│  Evaluation  │   │
│   │  (best.pt)   │   若新模型胜率 > 55%      │  (Arena 对弈)│   │
│   └──────────────┘                          └──────────────┘   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

每轮（iteration）：
1. 用当前最新模型生成 $N_{\text{games}}$ 局自对弈棋
2. 把棋谱加入数据缓冲区（FIFO，超出 50k 丢弃最早的）
3. 从缓冲区随机采样 $N_{\text{batches}}$ 个 batch，SGD 更新网络
4. 每 $K$ 轮做一次 Arena 评估：新模型 vs 上一版 best，若胜率 > 55% 则更新 best

## 2. 自对弈数据生成

### 2.1 单局自对弈流程

```python
def self_play_game(model, mcts, board_size, config):
    """Generate one self-play game, return list of training samples."""
    state = GameState(board_size=board_size, config=config)
    trajectory = []

    while not state.is_terminal():
        # MCTS search
        root = mcts.search(state)
        pi = mcts.get_action_distribution(root, temperature=temp_schedule(state.step, state.total_steps_est))

        # Record sample (state, policy, value=placeholder)
        trajectory.append({
            'state': state.clone(),
            'policy': pi,
            'player': state.current_player,
        })

        # Sample action (training: stochastic; eval: deterministic)
        action_idx = np.random.choice(len(pi), p=pi)
        move = index_to_move(action_idx, board_size)
        state.apply_move(move)

    # Backfill terminal value
    winner = state.winner()  # 'X', 'O', or 'draw'
    samples = []
    for entry in trajectory:
        if winner == 'draw':
            z = 0.0
        elif winner == entry['player']:
            z = +1.0
        else:
            z = -1.0
        samples.append((entry['state'], entry['policy'], z))
    return samples
```

### 2.2 温度调度

```python
def temp_schedule(step, total_steps):
    """High temperature early (explore), low late (exploit)."""
    ratio = step / max(total_steps, 1)
    if ratio < 0.3:
        return 1.0
    elif ratio < 0.7:
        return 0.5
    else:
        return 0.0   # Greedy in endgame
```

终局阶段（剩下 ≤ 5 步）始终用 $T=0$，避免无意义的随机落子。

### 2.3 自对弈并发

CPU 上用 `multiprocessing` 并行：

```python
from multiprocessing import Pool

def parallel_selfplay(model_path, n_games, n_workers, config):
    """Run n_games self-play games in parallel."""
    with Pool(n_workers) as pool:
        results = pool.starmap(
            _selfplay_worker,
            [(model_path, config) for _ in range(n_games)]
        )
    return [s for game in results for s in game]
```

每个 worker 独立加载模型副本，避免 GIL 竞争。8 核 CPU 上 1 局 6×6 自对弈约 30 秒（400 simulations/step），8 worker 并发约 4 局/分钟。

## 3. 数据缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, samples):
        """Add samples (list of (state, policy, value))."""
        for s in samples:
            if len(self.buffer) < self.capacity:
                self.buffer.append(s)
            else:
                self.buffer[self.pos] = s
                self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """Random sample batch_size samples."""
        idx = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idx]
        states  = torch.stack([state_to_tensor(s) for s, _, _ in batch])
        policies = torch.tensor(np.stack([p for _, p, _ in batch]), dtype=torch.float32)
        values  = torch.tensor([[v] for _, _, v in batch], dtype=torch.float32)
        return states, policies, values

    def __len__(self):
        return len(self.buffer)
```

### 3.1 缓冲区大小选择

- **太小** (<10k)：旧数据被快速丢弃，训练不稳定
- **太大** (>200k)：旧策略数据污染新模型
- **50k** 适合 6×6 棋盘，约 1500 局棋的样本量

### 3.2 数据增强

每次采样时随机应用 8 种对称变换（见 `docs/02_feature_planes.md` 第 5 节），等效 8 倍数据：

```python
def sample_with_augmentation(buffer, batch_size):
    states, policies, values = buffer.sample(batch_size)
    # Apply random D4 symmetry to each sample
    for i in range(len(states)):
        k = np.random.randint(4)
        flip = np.random.rand() < 0.5
        states[i] = apply_symmetry(states[i], k, flip)
        policies[i] = apply_symmetry_policy(policies[i], k, flip, board_size)
    return states, policies, values
```

## 4. 训练 Loss

$$
\mathcal{L} = \underbrace{-\sum_a \pi_a \log p_a}_{\text{policy cross-entropy}} + \underbrace{c_v (v - z)^2}_{\text{value MSE}} + \underbrace{c_{\ell2} \|\theta\|_2^2}_{\text{L2 reg}}
$$

默认 $c_v = 1.0$，$c_{\ell2} = 10^{-4}$。

```python
def compute_loss(model, batch, l2_coef=1e-4):
    states, target_policies, target_values = batch
    states = states.to(device)
    target_policies = target_policies.to(device)
    target_values = target_values.to(device)

    policy_logits, value = model(states)

    # Policy: cross-entropy with soft targets (MCTS distribution)
    log_policy = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -(target_policies * log_policy).sum(dim=-1).mean()

    # Value: MSE
    value_loss = F.mse_loss(value, target_values)

    # L2 regularization
    l2 = sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)
    l2_loss = l2_coef * l2

    return policy_loss + value_loss + l2_loss, policy_loss.item(), value_loss.item()
```

### 4.1 为什么 Policy 用 cross-entropy 而不是 KL？

MCTS 分布 $\pi$ 是 soft target（多个动作都有非零概率）。Cross-entropy：

$$-\sum_a \pi_a \log p_a$$

等价于最小化 $KL(\pi \| p)$ 减去常数 $\sum \pi \log \pi$（熵），所以数值上和 KL 差一个常数。Cross-entropy 计算更稳定（不需要 log π）。

## 5. 优化器

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-2,
    momentum=0.9,
    weight_decay=1e-4,   # L2 reg via weight_decay
    nesterov=True,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_iterations, eta_min=1e-4
)
```

### 5.1 为什么 SGD 不 Adam？

KataGo 和 AlphaZero 都用 SGD+momentum。原因：

1. RL 自对弈数据分布会变（non-stationary），Adam 的二阶矩估计可能不稳定
2. SGD+momentum 的"惯性"有助于跨越 policy 的局部最优
3. 长时间训练 SGD 收敛到更平的极小值（更稳健）

如果训练不稳定，可改用 Adam (lr=3e-4) 短时间快速验证。

### 5.2 学习率调度

- 初始 $10^{-2}$，余弦退火到 $10^{-4}$
- 训练 1000 iterations 后接近 $10^{-4}$
- 若训练后期 loss 不再下降，可手动降到 $10^{-5}$ 微调

## 6. 训练循环

```python
def train(model, buffer, config):
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)

    for iteration in range(config.iterations):
        # --- 1. Self-play ---
        model.eval()
        new_samples = parallel_selfplay(
            model_path=config.checkpoint_dir / f'model_iter{iteration}.pt',
            n_games=config.selfplay_games_per_iter,
            n_workers=config.n_workers,
            config=config,
        )
        buffer.push(new_samples)
        log.info(f'Iter {iteration}: generated {len(new_samples)} samples, buffer size {len(buffer)}')

        # --- 2. Training ---
        model.train()
        for step in range(config.train_batches_per_iter):
            batch = buffer.sample_with_augmentation(config.batch_size)
            loss, p_loss, v_loss = compute_loss(model, batch, config.l2_coef)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if step % 10 == 0:
                log.info(f'  step {step}: loss={loss.item():.4f}, p={p_loss:.4f}, v={v_loss:.4f}')
        scheduler.step()

        # --- 3. Checkpoint ---
        save_checkpoint(model, config.checkpoint_dir / f'model_iter{iteration+1}.pt')

        # --- 4. Evaluate every K iters ---
        if (iteration + 1) % config.eval_every == 0:
            arena_evaluate(model, config)

    save_checkpoint(model, config.checkpoint_dir / 'model_final.pt')
```

## 7. Arena 评估

新模型 vs 当前 best，胜率 > 55% 才更新 best：

```python
def arena_evaluate(new_model, best_model_path, config):
    """Play N games between new and best, return win rate of new."""
    wins = 0
    for game_idx in range(config.arena_games):
        # Alternate colors
        if game_idx % 2 == 0:
            x_player, o_player = new_model, best_model_path
        else:
            x_player, o_player = best_model_path, new_model
        winner = play_match(x_player, o_player, config)
        # Win from new_model's perspective
        new_is_x = (x_player is new_model)
        if (winner == 'X' and new_is_x) or (winner == 'O' and not new_is_x):
            wins += 1
    win_rate = wins / config.arena_games
    log.info(f'Arena: new vs best win_rate = {win_rate:.2%}')

    if win_rate > 0.55:
        save_checkpoint(new_model, best_model_path)
        log.info('New model accepted as best.')
        return True
    return False
```

### 7.1 评估时的 MCTS 配置

- `n_simulations`：评估时 **加倍**（如训练 200，评估 400），更精确
- `temperature`：始终 0.0（贪婪）
- `dirichlet_noise`：关闭

## 8. 超参数表（默认配置）

完整配置见 `configs/default.yaml`，关键项：

| 类别 | 参数 | 默认值 |
|------|------|--------|
| 棋盘 | `board_size` | 6 |
| 棋盘 | `initial_health` | 2 |
| 棋盘 | `attack_power` | 1 |
| 棋盘 | `heal_power` | 1 |
| 棋盘 | `diag_heal` | true |
| 棋盘 | `diag_attack` | true |
| 网络 | `in_channels` | 11 |
| 网络 | `channels` | 64 |
| 网络 | `n_blocks` | 6 |
| MCTS | `n_simulations_train` | 200 |
| MCTS | `n_simulations_eval` | 400 |
| MCTS | `c_puct` | 1.5 |
| MCTS | `dirichlet_alpha` | 0.3 |
| MCTS | `dirichlet_epsilon` | 0.25 |
| 训练 | `iterations` | 100 |
| 训练 | `selfplay_games_per_iter` | 50 |
| 训练 | `train_batches_per_iter` | 100 |
| 训练 | `batch_size` | 64 |
| 训练 | `lr` | 0.01 |
| 训练 | `weight_decay` | 1e-4 |
| 训练 | `lr_min` | 1e-4 |
| 缓冲 | `buffer_capacity` | 50000 |
| 评估 | `eval_every` | 5 |
| 评估 | `arena_games` | 20 |
| 评估 | `win_rate_threshold` | 0.55 |

## 9. 监控指标

训练过程记录到 TensorBoard（或简单 CSV），关注：

| 指标 | 健康范围 | 异常处理 |
|------|----------|----------|
| policy loss | 1.5 → 0.5 | 不降反升：lr 太大 / 数据分布漂移 |
| value loss | 0.6 → 0.2 | 卡在 0.5+：网络太小 / value 信号弱 |
| 自对弈局长 | 30 ± 5 步 | >50：模型不会下终局；<15：太快崩溃 |
| 自对弈 X 胜率 | 45%-55% | 偏向 >60%：先手优势过大，考虑加让子 |
| Arena 胜率 | 50%-60% | 一直 50%：训练停滞，加 lr 重启 |
| Buffer 多样性 | 100+ 种局面/100局 | <30：模型坍塌到固定开局 |

## 10. 课程学习 (Curriculum Learning)

为支持可调棋盘 5/6/7，采用课程学习：

```
Phase 1 (iter 0-30):   board_size=5, n_simulations=100
Phase 2 (iter 30-70):  board_size=6, n_simulations=200
Phase 3 (iter 70-100): board_size=7, n_simulations=400
```

每个 Phase 切换时：
- 学习率重置为初始值
- 缓冲区清空（旧棋盘样本不适用新棋盘）
- 检查点单独保存（`model_phase1.pt`, `model_phase2.pt`, ...）

完整训练脚本见 `python/train.py`。

## 11. KataGo 借鉴的关键训练技巧

1. **Self-play 数据要够多**：每轮至少 50 局，缓冲 50k
2. **温度退火**：训练后期降温，policy 更确定
3. **L2 + momentum**：SGD+momentum 比 Adam 在 RL 更稳
4. **梯度裁剪**：`clip_grad_norm_ = 1.0`，防止连锁 value 梯度爆炸
5. **Arena 评估**：不让训练 loss 主导，用真实对弈胜率判断模型好坏
6. **检查点频繁保存**：每轮存一次，便于回滚
7. **SWA (Stochastic Weight Averaging)**：训练后期（最后 20% iterations）平均权重，提升稳健性（本框架 v1 可选）

## 12. CPU 训练预期

| 配置 | 单 iter 时长 | 100 iter 总时长 |
|------|-------------|----------------|
| 1 worker, 200 sim/step | ~30 min | ~50 hr |
| 8 workers, 200 sim/step | ~5 min | ~8 hr |
| 8 workers, 100 sim/step | ~3 min | ~5 hr |

CPU 上跑 100 iter（约 8 小时）足以让模型对随机策略胜率 >85%，对原版 minimax(depth=2) 胜率 >40%。

迁移 GPU 后，单 iter 约 30 秒，100 iter 1 小时，可上 1000 iter 达到对 minimax(depth=3) 胜率 >50%。
