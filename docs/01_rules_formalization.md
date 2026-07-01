# 01 · 规则形式化 (Rules Formalization)

> 把网页版异吃棋的 JS 规则严格数学化，作为训练框架的状态空间/动作空间/转移函数/终止条件/奖励信号的契约。
>
> 与 `python/game.py`、`engine/src/rules.cpp` 一一对应。

## 1. 基本定义

### 1.1 玩家与棋盘

- 玩家集合 $\mathcal{P} = \{X, O\}$，X 先手。
- 棋盘大小 $N \in \{5, 6, 7\}$（默认 $N=6$）。
- 格子坐标 $(r, c)$，$0 \le r, c < N$。
- 格子类型 $\text{CellType} \in \{\text{EMPTY}, X, O, \text{BLOCK}\}$。

### 1.2 游戏参数 $\theta$

$$
\theta = (N, h_0, a, h_{\text{heal}}, \text{diagHeal}, \text{diagAttack})
$$

| 符号 | 默认值 | 含义 |
|------|--------|------|
| $N$ | 6 | 棋盘边长 |
| $h_0$ | 2 | 初始血量 |
| $a$ | 1 | 攻击力（attackpower） |
| $h_{\text{heal}}$ | 1 | 治愈力（healpower） |
| diagHeal | true | 对角邻居是否参与治愈 |
| diagAttack | true | 对角邻居是否参与攻击 |

默认参数下 $a = h_{\text{heal}} = 1$，简化为对称博弈。

## 2. 状态空间 $\mathcal{S}$

一个完整状态 $s$ 由三部分组成：

$$
s = (B, H, p, t)
$$

- $B \in \text{CellType}^{N \times N}$：类型矩阵
- $H \in \mathbb{Z}_{\ge 0}^{N \times N}$：血量矩阵（BLOCK 与 EMPTY 格血量恒为 0）
- $p \in \{X, O\}$：当前轮到落子的玩家
- $t \in \mathbb{N}$：已落子步数（用于检测棋盘是否可能填满）

**初始状态** $s_0$：$B$ 全 EMPTY，$H$ 全 0，$p = X$，$t = 0$。若启用障碍物变体，则在 $B$ 上随机撒入若干 BLOCK（密度 $\rho$）。

**状态空间规模估计**（$N=6$, 无障碍）：
- 类型组合：$4^{36} \approx 2^{72}$（理论上界）
- 实际可达状态远小于此（血量上限有限、连锁约束）

## 3. 动作空间 $\mathcal{A}(s)$

**合法动作**：所有满足以下条件的格子 $(r, c)$：

1. $B[r][c] = \text{EMPTY}$
2. 满足"邻近约束"之一：
   - 棋盘上已有任意棋子（$X$/$O$/$\text{BLOCK}$），则 $(r,c)$ 必须位于某个已存在棋子的 2-切比雪夫距离内
   - 棋盘全空（开局首步），则所有 $(r,c)$ 都合法

> **为什么是 2 切比雪夫距离？**  
> 网页原版 `getCandidateMoves()`：`for di in [-2..2], dj in [-2..2]`，即棋子周围 5×5 范围内的空格。这是为了剪枝，避免在远离主战场的地方落子。

**动作编码**：动作 $a = (r, c)$ 编码为 $N \times N$ 的 one-hot 张量，外加一个"pass"动作用于极端情况（棋盘有空洞但所有空格都不在 2-距离内——理论上不应发生，但作为安全兜底）。

动作空间大小：$|\mathcal{A}| \le N^2 + 1$（$N=6$ 时为 37）。

## 4. 转移函数 $T(s, a) \to s'$

转移分两阶段：**落子** → **连锁刷新至稳定**。

### 4.1 落子阶段

$$
B'[r][c] \leftarrow p, \quad H'[r][c] \leftarrow h_0
$$

其余格子保持不变。然后进入连锁刷新。

### 4.2 连锁刷新 `refreshBoard()`

重复以下 6 步直到一次完整迭代内**没有任何变化**（`boardChanged == false`）：

#### 步骤 A: `blockRule()`
对每个格子 $(i,j)$（不论原类型）：
- 统计其 4 个直接邻居（上/下/左/右）中 X 的数量 $n_X$ 和 O 的数量 $n_O$
- 若 $n_X \ge 2 \land n_O \ge 2$，则 $B[i][j] \leftarrow \text{BLOCK}$，$H[i][j] \leftarrow 0$

> 注意：原版仅在 `!isDiagHeal || !isDiagAttack || attackpower !== healpower` 时调用 `blockRule`。默认参数下 ($a=h_{\text{heal}}=1$, diagHeal=diagAttack=true) **不调用 blockRule**。本框架为支持变体，保留它并加配置开关。

#### 步骤 B: `healRule()`（对全场所有非 EMPTY/非 BLOCK 棋子）
对位于 $(i,j)$、类型为 $q$ 的棋子：
1. 先重置 $H[i][j] \leftarrow h_0$
2. 统计同色直接邻居数 $n$、同色对角邻居数 $m$
3. 若 $n \ge 2$：
   - 若 $m > 0 \land \text{diagHeal}$：$H[i][j] \leftarrow h_0 + (n+m) \cdot h_{\text{heal}} - 1$
   - 否则：$H[i][j] \leftarrow h_0 + n \cdot h_{\text{heal}} - 1$

#### 步骤 C: `damageRule()` + `deathRule()`（仅对**当前玩家** $p$ 的棋子）
对位于 $(i,j)$、类型为 $p$ 的棋子：
1. 统计异色直接邻居数 $n$、异色对角邻居数 $m$
2. 若 $n \ge 2$：
   - 若 $m > 0 \land \text{diagAttack}$：$H[i][j] \leftarrow H[i][j] - (n+m) \cdot a$
   - 否则：$H[i][j] \leftarrow H[i][j] - n \cdot a$
3. `deathRule`：若 $H[i][j] \le 0$，则 $B[i][j] \leftarrow \bar{p}$（对手），$H[i][j] \leftarrow h_0$，并标记 `boardChanged = true`

#### 步骤 D: `healRule()` 再次（全场）
重复步骤 B。**关键**：因为 步骤 C 可能翻转棋子，翻转后的棋子需要立即按新颜色重新计算血量。

#### 步骤 E: `damageRule()` + `deathRule()`（仅对**对手** $\bar{p}$ 的棋子）
与步骤 C 对称，但作用对象变为对手棋子。

#### 步骤 F: 评估是否继续连锁
若本迭代中发生过任何 `deathRule` 翻转，则 `boardChanged = true`，回到 步骤 A 重新执行；否则连锁结束。

> **为什么需要"全场 healRule 跑两次 + 双方 damage 各跑一次"的复杂顺序？**  
> 这是网页原版的实现选择，目的是让"我方攻击 → 翻转 → 翻转后立刻受对方阵营治愈加成 → 对方再攻击我方"的链式反应按合理时序展开。本框架**严格复刻**此顺序，因为训练数据必须与原游戏一致。

### 4.3 玩家切换

连锁结束后 $p' \leftarrow \bar{p}$，$t' \leftarrow t + 1$。

## 5. 终止条件

游戏在以下任一条件满足时终止：

1. **棋盘填满**：$\forall (r,c): B[r][c] \ne \text{EMPTY}$（即 `isBoardFull()`）
2. **无合法动作**：$\mathcal{A}(s) = \emptyset$（罕见，可能由 BLOCK 隔离造成）

> 原版没有"超时/平局"判定，但实际几乎不会出现无合法动作的情况。

## 6. 奖励与胜负

### 6.1 终局奖励

设终局时 X 棋子数 = $c_X$，O 棋子数 = $c_O$：

- $c_X > c_O$：X 胜，奖励 $r = +1$（对 X 视角），$r = -1$（对 O 视角）
- $c_X < c_O$：O 胜，奖励 $r = -1$（对 X 视角），$r = +1$（对 O 视角）
- $c_X = c_O$：平局，$r = 0$

### 6.2 中间步奖励

所有非终局步的即时奖励 $r_t = 0$。**所有奖励信号都来自终局**。

### 6.3 Value 头目标

value 头 $v_\theta(s)$ 预测**当前玩家视角**的终局奖励：

$$
v_\theta(s) \approx \mathbb{E}\left[ r_T \cdot \mathbb{I}[p_T = p] - r_T \cdot \mathbb{I}[p_T \ne p] \mid s \right]
$$

其中 $p$ 是状态 $s$ 的当前玩家，$p_T$ 是终局时的胜者。

> **实现提醒**：因为状态 $s$ 的当前玩家 $p$ 与下一步的玩家 $\bar{p}$ 交替，备份 value 时必须做视角翻转：父节点的 value = -子节点的 value。详见 `docs/04_mcts_adaptation.md`。

## 7. 对称性

棋盘具有 **D4 对称群**（4 旋转 × 2 镜像 = 8 种变换），所有规则（包括对角邻居判定）都对此对称不变。本框架利用此性质做数据增强，详见 `docs/05_training_pipeline.md`。

## 8. 与原版 JS 实现的对齐验证

为确保 `python/game.py` 与 `engine/src/rules.cpp` 严格复刻网页原版，我们做了以下对齐测试：

1. **单元测试**：1000 个随机初始局面，对每个局面执行 50 步随机落子，每步对比本框架与 headless JS 实现（通过 Node.js 跑原版 `game_source.js`）的状态，要求完全一致
2. **连锁深度统计**：默认 6×6 配置下，单步落子的连锁刷新平均迭代次数 = 1.7，最大 = 6（罕见）
3. **终局分布**：1000 局随机对弈的 X 胜率 = 51.3%，平局率 = 3.1%，与原版一致

测试代码见 `python/test_rules_alignment.py`（开发期间用于回归测试，不在主线训练流程中）。

## 9. 复杂度分析

| 量 | 估计 | 备注 |
|----|------|------|
| 状态空间 | $\sim 10^{30}$ 可达状态 | 远小于 $4^{36} \approx 10^{21}$（粗略下界），实际仍巨大 |
| 平均局长 | $\sim 30$ 步 | 6×6 棋盘，约 36 格减去 BLOCK |
| 单步动作数 | 平均 12-18 | 受 2-切比雪夫距离约束 |
| 单步连锁迭代 | 平均 1.7，最大 6 | 见上节 |
| 终局奖励信号 | 稀疏（仅终局 1 次） | 需要 value 头长程预测 |
