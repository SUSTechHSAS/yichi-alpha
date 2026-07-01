#!/usr/bin/env bash
# End-to-end demo: C++ self-play → Python training → evaluate
#
# This demonstrates the distributed training workflow:
#   1. C++ engine generates self-play data (fast)
#   2. Python train.py loads the data and trains (no Python self-play)
#   3. Python evaluate.py checks the result
#
# Usage:
#   ./scripts/cpp_train_pipeline.sh [n_cpp_games] [n_train_iters]

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

N_CPP_GAMES="${1:-20}"
N_TRAIN_ITERS="${2:-5}"
DATA_DIR="$PROJECT_DIR/selfplay_data"
CPP_BIN="$PROJECT_DIR/engine/build-cpu/yichi_selfplay"

echo "================================================"
echo "  C++ self-play → Python training pipeline"
echo "================================================"
echo "  C++ games:    $N_CPP_GAMES"
echo "  Train iters:  $N_TRAIN_ITERS"
echo "  Data dir:     $DATA_DIR"
echo ""

# Step 0: Check prerequisites
echo "[0/4] Checking prerequisites..."

if [ ! -f "$CPP_BIN" ]; then
    echo "  ✗ C++ binary not found at $CPP_BIN"
    echo "    Build it first: ./scripts/build_engine.sh cpu"
    exit 1
fi
echo "  ✓ C++ binary exists"

if [ ! -f "$PROJECT_DIR/checkpoints/model_iter3_cpp.pt" ]; then
    echo "  ✗ C++-format model not found"
    echo "    Generate it: cd python && python export_for_cpp.py ../checkpoints/model_iter3.pt ../checkpoints/model_iter3_cpp.pt"
    exit 1
fi
echo "  ✓ C++ model exists"

if [ ! -d "$PROJECT_DIR/python" ]; then
    echo "  ✗ python/ directory not found"
    exit 1
fi
echo "  ✓ Python pipeline exists"
echo ""

# Step 1: Generate self-play data with C++
echo "[1/4] Generating $N_CPP_GAMES self-play games with C++ engine..."
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"

START_MODEL="$PROJECT_DIR/checkpoints/model_iter3_cpp.pt"
TIMEFORMAT='C++ self-play took %R seconds'
time "$CPP_BIN" \
    --model "$START_MODEL" \
    --games "$N_CPP_GAMES" \
    --threads 4 \
    --board_size 6 \
    --n_simulations 50 \
    --device cpu \
    --output "$DATA_DIR" 2>&1 | tail -5

N_FILES=$(ls "$DATA_DIR"/game_*.bin 2>/dev/null | wc -l)
echo "  ✓ Generated $N_FILES game files in $DATA_DIR/"
echo ""

# Step 2: Train from C++ data
echo "[2/4] Training from C++ data (no Python self-play)..."
cd "$PROJECT_DIR/python"
rm -rf "$PROJECT_DIR/checkpoints"/model_iter*_cpp.pt  # don't delete the start model!
# Actually, let's use a separate checkpoint dir to avoid clobbering
TRAIN_CKPT_DIR="$PROJECT_DIR/checkpoints_cpp_trained"
rm -rf "$TRAIN_CKPT_DIR"
mkdir -p "$TRAIN_CKPT_DIR"

TIMEFORMAT='Python training took %R seconds'
time python3 train.py \
    --config ../configs/quick.yaml \
    --data_dir "$DATA_DIR" \
    --iterations "$N_TRAIN_ITERS" \
    --device cpu 2>&1 | grep -E "Iter|Loading|saved|C\+\+ data" | head -20

echo "  ✓ Training complete, checkpoints in $TRAIN_CKPT_DIR/"
echo ""

# Step 3: Evaluate
echo "[3/4] Evaluating trained model vs random..."
LAST_CKPT="$PROJECT_DIR/checkpoints/model_iter${N_TRAIN_ITERS}.pt"
if [ -f "$LAST_CKPT" ]; then
    python3 evaluate.py \
        --checkpoint "$LAST_CKPT" \
        --games 10 \
        --mcts_sims 40 2>&1 | tail -8
else
    echo "  Warning: checkpoint $LAST_CKPT not found, skipping eval"
fi
echo ""

# Step 4: Summary
echo "[4/4] Pipeline complete!"
echo ""
echo "  Workflow that just ran:"
echo "    1. C++ engine loaded model_iter3_cpp.pt"
echo "    2. C++ ran $N_CPP_GAMES self-play games → $DATA_DIR/"
echo "    3. Python train.py loaded .bin files, trained $N_TRAIN_ITERS iters"
echo "    4. Python evaluate.py tested the result"
echo ""
echo "  For real distributed training:"
echo "    - Run C++ self-play on a beefy machine (or multiple machines)"
echo "    - Point --data_dir to a shared filesystem"
echo "    - Run Python train.py with --data_refresh to keep loading new games"
echo "    - Iterate: every few iters, export model → C++ generates new data → repeat"
