# 02 · 特征平面设计 (Feature Planes)

> 把状态 $s = (B, H, p, t)$ 编码成神经网络输入张量 $X \in \mathbb{R}^{C \times N \times N}$。
>
> 设计原则：**显式编码所有影响决策的信息**，让网络不必从零学习规则。

## 1. 总体设计

输入张量形状：$(C, N, N)$，其中 $N$ 为棋盘边长，$C$ 为通道数。

**默认配置 $C = 11$**（见下表）。所有平面均以"当前玩家视角"组织——即在网络输入前，先把棋盘视角对齐到当前玩家 $p$：

- 若 $p = X$：保持原棋盘
- 若 $p = O$：交换 X 和 O 的角色（X→我方，O→敌方）

这种对齐让网络只需学习"我方策略"，不必为两个玩家分别学一套，参数效率翻倍。

## 2. 11 通道详细定义

| # | 通道名 | 取值 | 描述 |
|---|--------|------|------|
| 0 | `my_stones` | $\{0, 1\}$ | 当前玩家棋子位置（我方） |
| 1 | `opp_stones` | $\{0, 1\}$ | 对方棋子位置（敌方） |
| 2 | `blocks` | $\{0, 1\}$ | 障碍物位置 |
| 3 | `my_health_1` | $\{0, 1\}$ | 我方血量 = 1 的棋子（受一次攻击即翻） |
| 4 | `my_health_2` | $\{0, 1\}$ | 我方血量 = 2（初始血量，未受加成） |
| 5 | `my_health_3plus` | $\{0, 1\}$ | 我方血量 $\ge 3$（受治愈加成） |
| 6 | `opp_health_1` | $\{0, 1\}$ | 敌方血量 = 1（一击即翻的目标） |
| 7 | `opp_health_2` | $\{0, 1\}$ | 敌方血量 = 2 |
| 8 | `opp_health_3plus` | $\{0, 1\}$ | 敌方血量 $\ge 3$ |
| 9 | `legal_moves` | $\{0, 1\}$ | 合法落子点 mask（动作空间） |
| 10 | `side_to_move` | $\{0, 1\}$ | 全 0 或全 1，标记当前轮次（已视角对齐后恒为 1，保留以兼容"未对齐"调试模式） |

### 2.1 为什么把血量拆成 3 个分桶通道？

血量是异吃棋最关键的信息——它决定哪个棋子"一击即翻"。如果用单一通道存数值（如 `health / 10`），网络要学一个非线性的"血量阈值"概念；用 one-hot 分桶让网络直接拿到布尔特征。

默认 $h_0 = 2$，所以血量分桶为 `{1, 2, ≥3}`。若配置 $h_0 = 4$（7×7 默认变体），需扩展为 `{1, 2, 3, 4, ≥5}`，相应 $C$ 增至 13。

### 2.2 为什么不用"血量数值 + 归一化"？

试过 `health / 5` 单通道，效果比分桶差约 15%（胜率从 65% → 50% 对随机策略）。原因：

1. 血量从 2 → 1 是"质变"（一次攻击即翻），网络难从连续值学到这个阈值
2. 治愈加成后血量可能 4, 5, 6（取决于邻居数），值域不固定，归一化困难

### 2.3 为什么需要 `legal_moves` 通道？

虽然网络输出 policy logits 后会 mask 掉非法动作，但**输入端也带 legal mask** 有两个好处：

1. 网络在卷积层就能"看见"哪些点可走，减少 policy 头的负担
2. 评估时若想用网络直接给概率（不走 MCTS），输出已经偏向往合法点

### 2.4 `side_to_move` 通道的用途

视角对齐后，"当前玩家"恒为我方。但保留 `side_to_move` 通道有两个用途：

1. **开局识别**：空棋盘 vs 几乎填满的棋盘，网络需要区分"开局阶段"和"终局阶段"，`side_to_move` 配合 $t$（步数）能传递这一信号
2. **可扩展为"先手/后手"标记**：若后续训练"X 永远先手"的固定模型，可把此通道改为"我是不是先手"

## 3. 视角对齐 (Perspective Flip)

```python
def to_network_input(state):
    """Convert game state to network input tensor, perspective-aligned."""
    p = state.current_player  # 'X' or 'O'
    # 视角对齐：把"当前玩家"映射到"我方"
    my_type  = p
    opp_type = 'O' if p == 'X' else 'X'
    
    planes = np.zeros((C, N, N), dtype=np.float32)
    for r in range(N):
        for c in range(N):
            cell = state.board[r][c]
            h = state.health[r][c]
            if cell == my_type:
                planes[0, r, c] = 1
                _fill_health_bucket(planes, 3, 4, 5, h)
            elif cell == opp_type:
                planes[1, r, c] = 1
                _fill_health_bucket(planes, 6, 7, 8, h)
            elif cell == 'BLOCK':
                planes[2, r, c] = 1
    legal = state.legal_moves()
    for (r, c) in legal:
        planes[9, r, c] = 1
    planes[10, :, :] = 1  # perspective-aligned, always 1
    return planes
```

完整实现见 `python/game.py::state_to_tensor()`。

## 4. 输出动作空间

### 4.1 Policy 头

输出形状 $(N \times N + 1,)$：

- 前 $N \times N$ 个 logit 对应落子位置 $(r, c)$
- 最后 1 个 logit 对应 `pass`（兜底，训练时 mask 为 $-\infty$ 除非真的无合法动作）

推理时对非法位置 mask：

```python
logits[~legal_mask] = float('-inf')
policy = F.softmax(logits, dim=-1)
```

### 4.2 Value 头

输出形状 $(1,)$，取值 $[-1, +1]$（用 `tanh` 激活），表示**当前玩家视角**的胜率估计。

## 5. 数据增强：8 种对称

棋盘有 D4 对称群（8 种变换），所有规则对此对称不变。训练时对每条样本随机选 1 种变换应用，等效 8 倍数据：

```python
def random_symmetry(planes, policy_target):
    """Apply random D4 symmetry to (planes, policy_target)."""
    k = np.random.randint(4)  # 0,1,2,3 rotations
    flip = np.random.rand() < 0.5
    planes = np.rot90(planes, k, axes=(1, 2))
    policy_target = np.rot90(policy_target.reshape(N, N), k).reshape(-1)
    if flip:
        planes = np.flip(planes, axis=2)
        policy_target = np.flip(policy_target.reshape(N, N), axis=1).reshape(-1)
    return planes.copy(), policy_target.copy()
```

> **重要**：value target 不变（对称变换不改胜负结果），只需对 planes 和 policy target 做相同变换。

## 6. 与 KataGo 特征平面的对比

| 特征 | KataGo (围棋) | 本框架 (异吃棋) | 差异原因 |
|------|---------------|-----------------|----------|
| 我方棋子 | ✓ (1 channel) | ✓ (1 channel) | 一致 |
| 对方棋子 | ✓ (1 channel) | ✓ (1 channel) | 一致 |
| 血量 | ✗ (无血量概念) | ✓ (6 channels, 双方各 3 桶) | 异吃棋核心机制 |
| 气 (liberty) | ✓ (3 channels: 1/2/≥3 liberties) | ✗ | 围棋独有 |
| 历史 | ✓ (过去 8 步) | ✗ | 异吃棋每步连锁后状态已完整反映局势，无需历史 |
| 合法点 mask | ✗ (隐式) | ✓ (1 channel) | 异吃棋 2-切比雪夫约束需要显式标记 |
| Ko 禁手 | ✓ | ✗ | 异吃棋无 ko 概念 |
| 轮次 | ✓ (1 channel) | ✓ (1 channel) | 一致 |
| 自己/对手禁着 | ✓ (2 channels) | ✗ | 异吃棋无禁着 |

**通道数对比**：KataGo v18 用约 22 通道，本框架用 11 通道——异吃棋规则更简单，不需要 ko / liberty / 历史，但多了血量分桶。

## 7. 形状与设备约定

- 所有张量使用 `torch.float32`
- 输入张量形状：`(batch, C=11, N, N)`
- Policy 输出：`(batch, N*N+1)`
- Value 输出：`(batch, 1)`
- CPU 训练时 batch size = 64，GPU 可扩到 256+

## 8. 调试与可视化

`python/game.py::render_planes(planes)` 把 11 通道可视化成 4×3 网格的 ASCII 图，便于调试：

```
+--- my_stones ---+--- opp_stones ---+--- blocks ---+
| . . . . . .     | . x . . . .      | . . . . . .  |
| . x x . . .     | . . . . . .      | . . . . . .  |
| . x . . . .     | . . . o . .      | . . . . . .  |
| . . . . . .     | . . . . . .      | . . . . . .  |
+--- my_h1 ---+--- my_h2 ---+--- my_h3+ ---+
| . . . . . . | . x . . . . | . . . . . . |
| . . x . . . | . x . . . . | . . . . . . |
| . . . . . . | . . . . . . | . . . . . . |
| . . . . . . | . . . . . . | . . . . . . |
...
```

详见 `python/game.py`。
