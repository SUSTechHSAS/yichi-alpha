#!/usr/bin/env bash
# Build the C++ engine for either CPU or CUDA LibTorch.
#
# Usage:
#   ./scripts/build_engine.sh cpu      # default; uses /opt/libtorch-cpu
#   ./scripts/build_engine.sh cuda     # uses /opt/libtorch-cuda
#   ./scripts/build_engine.sh cpu /custom/path/to/libtorch
#   ./scripts/build_engine.sh cuda /custom/path/to/libtorch-cuda
#
# The script produces a binary at engine/build-{cpu,cuda}/yichi_selfplay
# with rpath embedded so it can be run directly without LD_LIBRARY_PATH.

set -e

MODE="${1:-cpu}"
CUSTOM_PATH="${2:-}"

case "$MODE" in
    cpu)
        DEFAULT_PATH="/opt/libtorch-cpu/libtorch"
        BUILD_DIR="build-cpu"
        ;;
    cuda)
        DEFAULT_PATH="/opt/libtorch-cuda/libtorch"
        BUILD_DIR="build-cuda"
        ;;
    *)
        echo "Usage: $0 [cpu|cuda] [custom_libtorch_path]"
        echo "  cpu   — build with CPU-only LibTorch (default)"
        echo "  cuda  — build with CUDA-enabled LibTorch (requires CUDA toolkit)"
        exit 1
        ;;
esac

LIBTORCH_PATH="${CUSTOM_PATH:-$DEFAULT_PATH}"
ENGINE_DIR="$(cd "$(dirname "$0")/.." && pwd)/engine"

echo "Building YichiAlpha C++ engine ($MODE mode)..."
echo "  Engine dir:  $ENGINE_DIR"
echo "  LibTorch:    $LIBTORCH_PATH"
echo "  Build dir:   $ENGINE_DIR/$BUILD_DIR"
echo ""

if [ ! -d "$LIBTORCH_PATH" ]; then
    echo "Error: LibTorch not found at $LIBTORCH_PATH"
    echo ""
    echo "Install $MODE LibTorch with:"
    if [ "$MODE" = "cpu" ]; then
        echo "  wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.12.1%2Bcpu.zip"
        echo "  sudo mkdir -p /opt/libtorch-cpu"
        echo "  sudo unzip libtorch-shared-with-deps-2.12.1+cpu.zip -d /opt/libtorch-cpu"
    else
        echo "  wget https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.12.1%2Bcu121.zip"
        echo "  sudo mkdir -p /opt/libtorch-cuda"
        echo "  sudo unzip libtorch-shared-with-deps-2.12.1+cu121.zip -d /opt/libtorch-cuda"
        echo ""
        echo "Note: CUDA build also requires NVIDIA CUDA Toolkit 12.1+ to be installed."
        echo "      Verify with: nvcc --version"
    fi
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "Error: cmake not installed"
    echo "  sudo apt install cmake"
    exit 1
fi

BUILD_PATH="$ENGINE_DIR/$BUILD_DIR"
mkdir -p "$BUILD_PATH"
cd "$BUILD_PATH"

echo "Configuring with CMake..."
cmake -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" \
      -DCMAKE_BUILD_TYPE=Release \
      "$ENGINE_DIR"

echo ""
echo "Compiling..."
make -j"$(nproc)"

echo ""
echo "✓ Build complete ($MODE mode)"
echo "  Binary: $BUILD_PATH/yichi_selfplay"
echo ""
echo "Verify it runs (rpath is embedded, no LD_LIBRARY_PATH needed):"
echo "  $BUILD_PATH/yichi_selfplay --help"
echo ""
echo "Or if you see 'libc10.so not found', set LD_LIBRARY_PATH as fallback:"
echo "  export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:\$LD_LIBRARY_PATH"
echo "  $BUILD_PATH/yichi_selfplay --help"
