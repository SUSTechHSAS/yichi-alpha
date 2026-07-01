#!/usr/bin/env bash
# Run a quick CPU training pipeline verification.
# Uses configs/quick.yaml (smaller network, fewer sims) so it fits in ~5 min.

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR/python"

# Clean previous runs (optional)
if [ "$1" == "--clean" ]; then
    echo "Cleaning previous checkpoints and logs..."
    rm -rf "$PROJECT_DIR/checkpoints"/* "$PROJECT_DIR/logs"/*
fi

echo "=== YichiAlpha quick training (8 iterations, CPU) ==="
python3 train.py --config "$PROJECT_DIR/configs/quick.yaml" --iterations 8

echo ""
echo "=== Evaluate final model vs random ==="
python3 evaluate.py \
    --checkpoint "$PROJECT_DIR/checkpoints/model_final.pt" \
    --games 8 --mcts_sims 60

echo ""
echo "=== Compare iter0 vs final (progression) ==="
python3 eval_progression.py --games 4 --mcts_sims 40

echo ""
echo "Done. Checkpoints in: $PROJECT_DIR/checkpoints/"
echo "             Logs in: $PROJECT_DIR/logs/"
