# YichiAlpha C++ Engine — CMake build
# =====================================
# Requires LibTorch (CPU or CUDA — same source code, different libtorch download).
#
# Install LibTorch (CPU):
#   wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.12.1%2Bcpu.zip
#   unzip libtorch-*.zip -d /opt/libtorch-cpu
#
# Install LibTorch (CUDA 12.1):
#   wget https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.12.1%2Bcu121.zip
#   unzip libtorch-*.zip -d /opt/libtorch-cuda
#
# Build (CPU):
#   mkdir build-cpu && cd build-cpu
#   cmake -DCMAKE_PREFIX_PATH=/opt/libtorch-cpu/libtorch ..
#   make -j$(nproc)
#   ./yichi_selfplay --help    # works directly, rpath embedded
#
# Build (CUDA) — use a SEPARATE build dir:
#   mkdir build-cuda && cd build-cuda
#   cmake -DCMAKE_PREFIX_PATH=/opt/libtorch-cuda/libtorch ..
#   make -j$(nproc)
#
# The CPU and CUDA builds MUST use separate build directories because CMake
# caches the libtorch path in CMakeCache.txt — mixing them will cause link errors.

cmake_minimum_required(VERSION 3.18)
project(yichi_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# LibTorch
find_package(Torch REQUIRED)

# OpenMP (for parallel self-play)
find_package(OpenMP REQUIRED)

add_executable(yichi_selfplay
    src/board.cpp
    src/rules.cpp
    src/mcts.cpp
    src/neuralnet.cpp
    src/selfplay_main.cpp
)

target_include_directories(yichi_selfplay PRIVATE include)
target_link_libraries(yichi_selfplay PRIVATE
    "${TORCH_LIBRARIES}"
    OpenMP::OpenMP_CXX
)

# Embed rpath so the binary finds libtorch libs without LD_LIBRARY_PATH.
# This makes ./yichi_selfplay work standalone after build.
# We use the absolute libtorch lib path captured at configure time.
if(TORCH_INSTALL_PREFIX)
    set_target_properties(yichi_selfplay PROPERTIES
        BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
        INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Enable optimizations
target_compile_options(yichi_selfplay PRIVATE
    -O3
    -march=native
    -ffast-math
    -Wall
    -Wextra
)

# Detect CUDA support in this libtorch build
if(DEFINED CAFFE2_USE_CUDA OR TORCH_CUDA)
    message(STATUS "  CUDA support: ENABLED")
else()
    # Heuristic: check if libtorch_cuda.so exists in the lib dir
    if(EXISTS "${TORCH_INSTALL_PREFIX}/lib/libtorch_cuda.so")
        message(STATUS "  CUDA support: ENABLED (detected libtorch_cuda.so)")
        target_compile_definitions(yichi_selfplay PRIVATE YICHI_USE_CUDA=1)
    else()
        message(STATUS "  CUDA support: disabled (CPU-only build)")
    endif()
endif()

# Print build info
message(STATUS "YichiAlpha engine configuration:")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  LibTorch: ${TORCH_INSTALL_PREFIX}")
message(STATUS "  OpenMP: ${OpenMP_CXX_VERSION}")
message(STATUS "  RPATH: ${TORCH_INSTALL_PREFIX}/lib (embedded in binary)")

