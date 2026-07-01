#!/usr/bin/env bash
# Build the C++ engine. Requires CMake + LibTorch.
#
# Usage:
#   ./scripts/build_engine.sh [libtorch_path]
#
# Default libtorch_path: /opt/libtorch

set -e

LIBTORCH_PATH="${1:-/opt/libtorch}"
ENGINE_DIR="$(cd "$(dirname "$0")/.." && pwd)/engine"

echo "Building YichiAlpha C++ engine..."
echo "  Engine dir: $ENGINE_DIR"
echo "  LibTorch:   $LIBTORCH_PATH"

if [ ! -d "$LIBTORCH_PATH" ]; then
    echo "Error: LibTorch not found at $LIBTORCH_PATH"
    echo "Install it with:"
    echo "  wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.12.0%2Bcpu.zip"
    echo "  unzip libtorch-*.zip -d /opt/"
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "Error: cmake not installed"
    echo "  sudo apt install cmake"
    exit 1
fi

BUILD_DIR="$ENGINE_DIR/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" -DCMAKE_BUILD_TYPE=Release "$ENGINE_DIR"
make -j"$(nproc)"

echo ""
echo "Build complete. Binary: $BUILD_DIR/yichi_selfplay"
echo ""
echo "Example usage:"
echo "  $BUILD_DIR/yichi_selfplay \\"
echo "    --model ../checkpoints/model_iter7.pt \\"
echo "    --games 100 --threads 8 --board_size 6 \\"
echo "    --output ./selfplay_data/"
