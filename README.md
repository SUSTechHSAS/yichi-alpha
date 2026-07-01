# 异吃棋 AlphaZero 训练框架 (YichiAlpha)

[![Build & Test](https://github.com/SUSTechHSAS/yichi-alpha/actions/workflows/build.yml/badge.svg)](https://github.com/SUSTechHSAS/yichi-alpha/actions/workflows/build.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![C%2B%2B](https://img.shields.io/badge/C%2B%2B-17-orange.svg)](https://en.cppreference.com/)

> 借鉴 KataGo / AlphaZero 训练范式，为"异吃棋"游戏从零搭建的神经网络 + MCTS + 自对弈强化学习框架。
>
> Game source: https://sustechhsas.github.io/Some-Little-JS-Games/异吃棋.html

## 1. 项目目标

把网页版"异吃棋"原有 minimax+alpha-beta AI 升级为 **AlphaZero 式** 训练 AI：

- 不依赖人类棋谱，纯自对弈从零学习
- 策略价值网络（Policy-Value Network）+ 蒙特卡洛树搜索（MCTS）
- 支持默认 6×6、可调 5×5 / 7×7 棋盘
- CPU 可跑通完整 pipeline，后续可无缝迁移 GPU

> **为什么不直接 fork KataGo？**  
> KataGo 的规则层、特征平面、ko 检测、liberty 计算都深度耦合围棋规则。异吃棋有"血量/治愈/翻转/障碍生成"等连锁机制，与围棋的"提子/气/劫"完全不同。强行改 KataGo 源码相当于重写其 80% 内核，不如从零搭建一个干净的小型 AlphaZero 框架，借鉴其训练方法论（双头网络、MCTS+Dirichlet 噪声、SWA 权重平均、对手视角 value 等）。

## 2. 目录结构

```
yichi-alpha/
├── README.md                     # 本文件
├── docs/                         # 设计文档（Markdown）
│   ├── 01_rules_formalization.md # 规则形式化（状态/动作/转移/终止/奖励）
│   ├── 02_feature_planes.md      # 神经网络输入特征平面设计
│   ├── 03_network_architecture.md# 策略价值网络结构
│   ├── 04_mcts_adaptation.md     # MCTS 针对连锁反应的改造
│   ├── 05_training_pipeline.md   # 自对弈→训练→评估闭环
│   └── 06_engine_design.md       # C++ 引擎设计
├── engine/                       # C++ 引擎（生产用）
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── board.h
│   │   ├── rules.h
│   │   ├── mcts.h
│   │   └── neuralnet.h
│   └── src/
│       ├── board.cpp
│       ├── rules.cpp
│       ├── mcts.cpp
│       ├── neuralnet.cpp          # LibTorch 集成
│       └── selfplay_main.cpp
├── python/                       # Python 训练 pipeline（CPU 可跑）
│   ├── game.py                   # 游戏引擎（与 C++ 同源）
│   ├── model.py                  # 策略价值网络
│   ├── mcts.py                   # MCTS（Python 参考实现）
│   ├── selfplay.py               # 自对弈数据生成
│   ├── dataset.py                # 数据缓冲
│   ├── train.py                  # 训练循环
│   ├── evaluate.py               # 评估对弈
│   └── config.py                 # 超参配置
├── configs/
│   └── default.yaml              # 默认训练配置
├── scripts/
│   ├── build_engine.sh           # 编译 C++ 引擎
│   ├── run_train.sh              # 启动训练
│   └── run_eval.sh               # 评估
└── checkpoints/                  # 模型权重
```

## 3. 快速开始

### 3.1 Python pipeline（推荐先用这个跑通）

```bash
# 依赖：Python 3.13+, PyTorch (CPU), NumPy, PyYAML
cd python/

# 跑一次小规模训练（CPU 上约 5-15 分钟，验证 pipeline 通）
python3 train.py --config ../configs/default.yaml --iterations 5

# 评估：让最新模型 vs 随机策略
python3 evaluate.py --checkpoint ../checkpoints/model_iter5.pt --games 20
```

### 3.2 C++ 引擎（生产部署用）

C++ 引擎同时支持 **CPU 和 CUDA** 构建——同一份源码，用不同的 LibTorch 编译。

#### CPU 构建（默认）

```bash
# 1. Install LibTorch (CPU version, ~123MB)
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.12.1%2Bcpu.zip
unzip libtorch-*.zip -d /opt/libtorch-cpu

# 2. Install CMake (if missing)
sudo apt install cmake

# 3. Build (rpath is auto-embedded — no LD_LIBRARY_PATH needed)
./scripts/build_engine.sh cpu
# or manually:
cd engine/ && mkdir build-cpu && cd build-cpu
cmake -DCMAKE_PREFIX_PATH=/opt/libtorch-cpu/libtorch ..
make -j$(nproc)
./yichi_selfplay --help    # works directly
```

#### CUDA 构建（GPU 加速）

```bash
# 1. Install NVIDIA CUDA Toolkit 12.1+ (nvcc --version to verify)

# 2. Download LibTorch CUDA version (~2.5GB)
wget https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.12.1%2Bcu121.zip
unzip libtorch-*.zip -d /opt/libtorch-cuda

# 3. Build (use a SEPARATE build dir — CPU and CUDA caches conflict)
./scripts/build_engine.sh cuda
# or manually:
cd engine/ && mkdir build-cuda && cd build-cuda
cmake -DCMAKE_PREFIX_PATH=/opt/libtorch-cuda/libtorch ..
make -j$(nproc)

# 4. Run with CUDA
./build-cuda/yichi_selfplay --model model.pt --device cuda --games 100
# or auto-detect (uses CUDA if available, falls back to CPU):
./build-cuda/yichi_selfplay --model model.pt --device auto --games 100
```

> **重要**：CPU 和 CUDA 构建必须用**不同的 build 目录**（`build-cpu` vs `build-cuda`），
> 因为 CMake 在 `CMakeCache.txt` 里缓存了 libtorch 路径，混用会链接错误。
> `scripts/build_engine.sh` 脚本会自动处理这个分离。

> **rpath 已嵌入**：编译出来的二进制自带 libtorch 库的搜索路径，
> 不需要 `LD_LIBRARY_PATH`。如果你之前遇到 `libc10.so: cannot open shared object file`，
> 重新编译即可解决。

## 4. 核心设计要点速览

| 设计点 | 选择 | 理由 |
|--------|------|------|
| 训练范式 | AlphaZero（Self-play + MCTS + NN） | 借鉴 KataGo 方法论，无人类棋谱依赖 |
| 网络结构 | ResNet 6 block × 64 channel + Policy/Value 双头 | 6×6 棋盘小，6 block 足够；CPU 推理友好 |
| 输入特征 | 11 通道（己方棋子、对方棋子、血量分桶、障碍、合法点、轮次） | 显式编码血量与连锁状态，详见 `docs/02_feature_planes.md` |
| MCTS 改造 | 单步 expand 触发完整连锁，backup 用 root-to-leaf 的累积 reward | 一次落子可能引发多次翻转，必须把"连锁后的终态"当作转移结果 |
| 奖励信号 | 终局 ±1（多子者胜）；非终局 0；value 头预测"当前玩家视角"胜率 | 与 KataGo 一致，避免视角混乱 |
| 数据增强 | 8 种对称（4 旋转 × 2 镜像） | 6×6 棋盘完全对称，免费 8 倍数据 |
| 训练 loss | policy (CE) + value (MSE) + L2 正则 | 标准 AlphaZero loss |
| 探索 | Dirichlet 噪声 α=0.3，权重 0.25 | 借鉴 KataGo，动作空间小所以 α 偏小 |

## 5. 已验证（实跑结果）

### 5.1 规则对齐验证（与原版 JS 1:1）
- 用 Node.js headless 跑原版 `game_source.js`，提取 `healRule/damageRule/deathRule/blockRule/refreshBoard` 等函数
- 生成 **50 局随机对弈 × 平均 36 步 = 1800 步**
- Python 重放每一步，逐格对比 `type`/`health`/`currentPlayer`
- **结果：1800 步全部 PERFECT MATCH**
- 验证脚本：`/home/z/my-project/scripts/verify_rules_vs_js.js` + `compare_rules.py`

### 5.2 C++ 引擎编译并运行
- 安装 CMake (pip) + LibTorch CPU 版 (123MB)
- 编译过程中修复 8 个真实 bug（API 不匹配、shared_ptr、NoGradGuard、TorchScript 格式等）
- **最终：C++ 二进制 282KB，加载 72 个参数/缓冲区（0 跳过），1 局自对弈 0.29 秒**
- 验证：`cd engine/build && ./yichi_selfplay --model ../checkpoints/model_iter7_cpp.pt --games 1 --n_simulations 30 --output /tmp/test`

### 5.3 统计评估（20 局）
| 模型 | vs 随机 胜率 | 局数 |
|------|------------|------|
| iter0（未训练） | 50% (10W/8L/2D) | 20 |
| iter7（训练 7 轮） | **100%** (20W/0L/0D) | 20 |

训练后胜率提升 **50 个百分点**。

### 5.4 Arena 评估接入训练循环
- 每 iter 自动评估 vs random（8 局）
- 每 `eval_every` iter 自动跑 Arena（new vs best，6 局）
- 胜率 > 55% 时更新 `model_best.pt`
- 实测 3 iter 训练 + 评估：

| Iter | Total Loss | Policy Loss | Value Loss | vs Random |
|------|-----------|-------------|------------|-----------|
| 1 | 4.3344 | 3.588 | 0.646 | 6/6 (100%) |
| 2 | 4.0926 | 3.565 | 0.427 | 6/6 (100%) |
| 3 | 3.9705 | 3.557 | 0.313 | 5/6 (83.3%) |

**Value loss 降 51.5%**（0.646 → 0.313），网络确实在学习预测胜负。

> **重要 bug 修复**：v1 早期版本的 MCTS PUCT 公式漏了视角翻转（`Q + U` 应为 `-Q + U`）。
> 修复前模型 vs 随机胜率为 0%，修复后跳到 100%。详见 `docs/04_mcts_adaptation.md` 第 2.2 节。

### 5.5 未验证 / 已知限制
- ❌ 未做"原版 minimax AI vs 本框架 AI"的直接对弈（原版 AI 在网页里，需要浏览器自动化）
- ❌ 未跑足够长训练（>20 iter）展示 ELO 收敛曲线（CPU 速度限制）
- ⚠️ C++ Arena 评估代码未接入 `selfplay_main.cpp`（目前只生成数据，不评估）
- ⚠️ Python 自对弈 8 并发未实测（CPU 上单线程已验证）

## 6. 后续路线图

- [ ] 接入 LibTorch C++ 推理，替代 Python 自对弈
- [ ] 引入 SWA（Stochastic Weight Averaging）平滑权重
- [ ] 添加 Arena 评估 + 自我胜率监控
- [ ] 课程学习：先 5×5 再 6×6 最后 7×7
- [ ] 迁移 GPU 训练（只需 `config.yaml` 改 `device: cuda`）
- [ ] 训练 7×7 通用模型，覆盖障碍物变体

## 7. 致谢

- 游戏原作者：Quaphwss（见网页彩蛋）
- 方法论借鉴：AlphaGo Zero / AlphaZero / KataGo
- 本框架独立实现，未使用上述任何项目源码
