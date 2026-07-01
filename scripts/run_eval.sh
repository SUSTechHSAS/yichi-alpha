#!/usr/bin/env bash
# Evaluate a trained model checkpoint.

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${1:-$PROJECT_DIR/checkpoints/model_final.pt}"
GAMES="${2:-20}"
SIMS="${3:-100}"

if [ ! -f "$CKPT" ]; then
    echo "Error: checkpoint not found at $CKPT"
    echo "Usage: $0 <checkpoint.pt> [n_games=20] [mcts_sims=100]"
    exit 1
fi

cd "$PROJECT_DIR/python"
python3 evaluate.py \
    --checkpoint "$CKPT" \
    --games "$GAMES" \
    --mcts_sims "$SIMS"
