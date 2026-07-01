# 04 · MCTS 改造 (MCTS Adaptation)

> 异吃棋最大的特殊性：**一次落子可能引发多步连锁反应**（伤害→翻转→再触发伤害→…）。标准 MCTS 假设"动作→下一个状态"是单步转移，本框架必须把"动作→连锁稳定后的状态"作为完整转移。

## 1. 标准 AlphaZero MCTS 回顾

每棵搜索树节点 $v$ 保存：

- $N(v)$：访问次数
- $W(v)$：累积 value
- $Q(v) = W(v) / N(v)$：均值 value
- $P(v)$：来自策略网络的先验概率
- 子节点列表（按合法动作展开）

**4 阶段循环**：

1. **Selection**：从根开始，按 PUCT 公式选子节点
   $$a_t = \arg\max_a \left[ Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)} \right]$$
2. **Expansion & Evaluation**：到达叶节点时，用网络前向得到 $(p, v)$，展开所有合法子节点
3. **Backup**：把 $v$（取负，因为视角翻转）沿路径回传累加
4. **Play**：根节点访问次数 $\ge$ 阈值后，按访问次数分布选动作

## 2. 异吃棋的关键改造点

### 2.1 转移 = "落子 + 连锁刷新至稳定"

标准 MCTS 的"动作"对应一步状态转移。异吃棋中：

```
状态 s -- 落子动作 a --> 中间状态 s' -- 连锁刷新 --> 稳定状态 s''
```

我们定义 **转移 $T(s, a) = s''$**（连锁稳定后的状态），MCTS 在展开子节点时**直接调用 `apply_move_with_chain()`**，把连锁刷新算入转移函数。

**实现层面**：
- `MCTSNode` 的 `expand()` 一次性执行完所有连锁
- 不需要在树里把连锁中间状态作为节点（那样会指数爆炸）
- 这与围棋"一次落子+一次提子"是同构的，只是异吃棋的"提子"可能多次级联

### 2.2 视角翻转的 Backup

value 头预测"当前玩家视角"的胜率。父节点是玩家 $p$，子节点是玩家 $\bar{p}$。Backup 时：

```
leaf_value (from network, perspective of leaf node's player)
  → negate when backing up to parent
  → negate again when backing up to grandparent
  → ...
```

具体伪代码：

```python
def backup(node, value):
    # value is from the perspective of `node.player`
    while node is not None:
        node.N += 1
        node.W += value
        node.Q = node.W / node.N
        value = -value   # flip perspective
        node = node.parent
```

### 2.3 Root 节点的 Dirichlet 噪声

AlphaZero 在 root 节点加 Dirichlet 噪声以鼓励探索：

$$P'(s, a) = (1 - \epsilon) \cdot P(s, a) + \epsilon \cdot \eta_a, \quad \eta \sim \text{Dir}(\alpha)$$

异吃棋动作空间小（$N=6$ 时 37），$\alpha$ 取 **0.3**（AlphaZero 围棋 $\alpha=0.03$ 是因为 362 动作空间），$\epsilon = 0.25$。

> **为什么 $\alpha$ 比围棋大？**  
> Dirichlet 分布的 $\alpha$ 越小越集中（少数动作获得大部分概率质量），越大越均匀。动作空间小时，需要更"散"的噪声才能覆盖所有合理动作。

### 2.4 虚拟损失 (Virtual Loss) 多线程

C++ 引擎多线程自对弈时，多个线程共享同一棵搜索树，用虚拟损失避免重复访问：

```cpp
// Thread A 访问节点 v：
v.virtual_loss += VL;        // 标记"已访问"
v.W -= VL;
v.Q = v.W / v.N;
// ... simulate ...
// Backup 完成后恢复：
v.W += VL;
v.virtual_loss -= VL;
```

Python 参考实现单线程，不使用虚拟损失。C++ 引擎在 `engine/src/mcts.cpp` 实现。

## 3. MCTSNode 数据结构

```python
class MCTSNode:
    __slots__ = ['state', 'parent', 'move', 'prior',
                 'children', 'N', 'W', 'Q', 'is_expanded']

    def __init__(self, state, parent=None, move=None, prior=0.0):
        self.state = state              # 游戏状态（已连锁稳定）
        self.parent = parent
        self.move = move                # (r, c) or 'pass'
        self.prior = prior              # P(s, a) from network
        self.children = {}              # move -> MCTSNode
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.is_expanded = False

    def ucb_score(self, c_puct):
        """PUCT: Q + U"""
        u = c_puct * self.prior * \
            (np.sqrt(self.parent.N) / (1 + self.N))
        return self.Q + u

    def best_child(self, c_puct):
        return max(self.children.values(),
                   key=lambda c: c.ucb_score(c_puct))
```

## 4. 完整搜索流程

```python
class MCTS:
    def __init__(self, model, c_puct=1.5, n_simulations=400,
                 dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
                 device='cpu'):
        self.model = model
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = device

    def search(self, root_state):
        root = MCTSNode(root_state.clone())
        policy_logits, _ = self.evaluate(root)
        policy = F.softmax(policy_logits, dim=-1).cpu().numpy()[0]

        # Expand root with Dirichlet noise
        legal_moves = root_state.legal_moves()
        self.expand(root, policy, legal_moves, add_noise=True)

        for _ in range(self.n_simulations):
            node = self.select(root)
            value = self.evaluate_and_expand(node)
            self.backup(node, value)

        return root

    def select(self, root):
        """Traverse tree by PUCT until leaf."""
        node = root
        while node.is_expanded and not node.state.is_terminal():
            node = node.best_child(self.c_puct)
        return node

    def evaluate_and_expand(self, node):
        """Network forward + expand leaf + return value (node's perspective)."""
        if node.state.is_terminal():
            # Terminal: return true reward from this node's perspective
            return node.state.terminal_reward_for_current_player()

        policy_logits, value = self.evaluate(node)
        policy = F.softmax(policy_logits, dim=-1).cpu().numpy()[0]
        legal_moves = node.state.legal_moves()
        self.expand(node, policy, legal_moves, add_noise=False)
        return value.item()

    def evaluate(self, node):
        """Run network on node.state, return (policy_logits, value)."""
        x = state_to_tensor(node.state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.model(x)
        return policy_logits, value

    def expand(self, node, policy, legal_moves, add_noise):
        """Create child nodes for all legal moves."""
        if add_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
        for i, move in enumerate(legal_moves):
            idx = move_to_index(move, node.state.board_size)
            p = policy[idx]
            if add_noise:
                p = (1 - self.dirichlet_epsilon) * p + \
                    self.dirichlet_epsilon * noise[i]
            # Apply move + chain reaction
            child_state = node.state.clone()
            child_state.apply_move(move)   # This runs the full chain internally
            node.children[move] = MCTSNode(
                state=child_state, parent=node, move=move, prior=p
            )
        node.is_expanded = True

    def backup(self, node, value):
        """Back up value with perspective flip."""
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value
            node = node.parent

    def get_action_distribution(self, root, temperature=1.0):
        """Return policy target pi ~ visits^1/T."""
        visits = np.zeros(root.state.board_size ** 2 + 1, dtype=np.float32)
        for move, child in root.children.items():
            idx = move_to_index(move, root.state.board_size)
            visits[idx] = child.N
        if temperature == 0:
            # Greedy: pick argmax
            result = np.zeros_like(visits)
            result[visits.argmax()] = 1.0
            return result
        visits = visits ** (1.0 / temperature)
        return visits / visits.sum()
```

## 5. 动作选择温度调度

训练早期需要更多探索（高温度），后期接近确定性（低温度）：

```python
def temperature_schedule(step, total_steps):
    if step < 0.3 * total_steps:
        return 1.0   # Exploration phase
    elif step < 0.7 * total_steps:
        return 0.5   # Balanced
    else:
        return 0.0   # Greedy (deterministic)
```

终局阶段（剩下 < 5 步）始终用 $T = 0$，因为已经接近终局，无需探索。

## 6. 关键超参数

| 参数 | 默认值 | 来源 | 调优建议 |
|------|--------|------|----------|
| `n_simulations` | 400 | AlphaZero Atari Go 200, Go 800 | CPU 慢可降到 100-200 |
| `c_puct` | 1.5 | AlphaGo Zero 1.0, KataGo 1.4 | 异吃棋 value 信号弱，略高更好 |
| `dirichlet_alpha` | 0.3 | AlphaZero Go 0.03, Chess 0.3 | 动作空间 37，介于 chess(4672) 和 tictactoe(9) 之间 |
| `dirichlet_epsilon` | 0.25 | AlphaZero 标准 | 0.2-0.3 都可 |
| `temperature` | 见调度 | AlphaZero 标准 | 训练后期用 0.0 |

## 7. 与标准 MCTS 的差异速查

| 维度 | 标准 MCTS (AlphaZero) | 本框架 |
|------|----------------------|--------|
| 转移函数 | 单步 | 落子 + 完整连锁 |
| 节点状态 | 落子后立即状态 | 连锁稳定后状态 |
| value 视角 | 当前玩家 | 当前玩家（一致） |
| backup | 取负传递 | 取负传递（一致） |
| Dirichlet α | 围棋 0.03, 国际象棋 0.3 | 0.3（同 chess） |
| 虚拟损失 | 多线程时用 | Python 不用，C++ 用 |
| 终局判断 | 网络预测 | 真实规则判定（更可靠） |
| 动作空间 | 19×19+1=362 | 6×6+1=37 |

## 8. 性能优化

### 8.1 批量推理

每次 `evaluate()` 只前向 1 个样本，CPU 浪费严重。优化：

- **Leaf Queue**：积累多个叶节点，批量前向
- **C++ 引擎**：用 `torch::Tensor::batch()` 一次前向 32-64 个叶节点

```python
# 批量推理版本（伪代码）
class BatchedMCTS:
    def search(self, root_state):
        root = MCTSNode(root_state.clone())
        leaves = []
        for _ in range(self.n_simulations):
            node = self.select(root)
            if node.state.is_terminal():
                self.backup(node, node.state.terminal_reward())
            else:
                leaves.append(node)
                if len(leaves) >= self.batch_size:
                    self.evaluate_batch(leaves)
                    leaves = []
        if leaves:
            self.evaluate_batch(leaves)
        return root

    def evaluate_batch(self, leaves):
        batch = torch.stack([state_to_tensor(l.state) for l in leaves])
        with torch.no_grad():
            policies, values = self.model(batch)
        for leaf, p, v in zip(leaves, policies, values):
            self.expand(leaf, ...)
            self.backup(leaf, v.item())
```

### 8.2 状态克隆复用

`MCTSNode.expand()` 里每个子节点都要 `clone()` 一次状态——这是主要开销。优化：

- **C++ 引擎**：用 `std::shared_ptr<Board>` + copy-on-write
- **Python**：用 `np.array.copy()`，已经足够快

### 8.3 重复局面缓存

异吃棋的连锁可能导致局面重复出现（虽然概率低）。在 root 的多次 simulation 中，可以缓存 `(state_hash, (policy, value))` 避免重复推理。

```python
class CachedMCTS(MCTS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_cache = {}

    def evaluate(self, node):
        h = node.state.hash()
        if h in self.eval_cache:
            return self.eval_cache[h]
        result = super().evaluate(node)
        self.eval_cache[h] = result
        return result
```

## 9. 训练数据的生成

每步 MCTS 后，记录一条训练样本 $(s_t, \pi_t, z_t)$：

- $s_t$：状态特征张量
- $\pi_t$：MCTS 访问次数分布（`get_action_distribution(temperature=1.0)`）
- $z_t$：终局奖励（从当前玩家视角，可能取负）

终局后，所有样本的 $z_t$ 都从最后结果反推：

```python
def collect_selfplay_data(model, state, mcts):
    """Run one self-play game, return list of (state, policy, value) tuples."""
    trajectory = []
    while not state.is_terminal():
        root = mcts.search(state)
        pi = mcts.get_action_distribution(root, temperature=...)
        # Sample action from pi (or argmax if T=0)
        action_idx = np.random.choice(len(pi), p=pi)
        move = index_to_move(action_idx, state.board_size)
        trajectory.append((state.clone(), pi, None))  # value filled later
        state.apply_move(move)

    # Backfill terminal value
    winner = state.winner()  # 'X', 'O', or 'draw'
    for i, (s, pi, _) in enumerate(trajectory):
        if winner == 'draw':
            z = 0.0
        elif winner == s.current_player:
            z = +1.0
        else:
            z = -1.0
        trajectory[i] = (s, pi, z)
    return trajectory
```

## 10. 总结

异吃棋的 MCTS 改造核心是 **"把连锁刷新纳入转移函数"**，其余部分与标准 AlphaZero MCTS 一致。这使得：

- 搜索树规模可控（不会因连锁爆炸）
- value 信号在"真实稳定状态"上评估，更可靠
- 可以直接复用 AlphaZero 的 PUCT/Dirichlet/backup 逻辑

后续若要进一步加速，主要方向是 **C++ 引擎 + 批量推理 + 多线程虚拟损失**，详见 `docs/06_engine_design.md`。
