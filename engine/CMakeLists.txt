# YichiAlpha C++ Engine — CMake build
# =====================================
# Requires LibTorch (CPU or CUDA).
#
# Install LibTorch (CPU):
#   wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.12.0%2Bcpu.zip
#   unzip libtorch-*.zip -d /opt/
#
# Build:
#   mkdir build && cd build
#   cmake -DCMAKE_PREFIX_PATH=/opt/libtorch ..
#   make -j$(nproc)

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

# Enable optimizations
target_compile_options(yichi_selfplay PRIVATE
    -O3
    -march=native
    -ffast-math
    -Wall
    -Wextra
)

# Print build info
message(STATUS "YichiAlpha engine configuration:")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  LibTorch: ${TORCH_INSTALL_PREFIX}")
message(STATUS "  OpenMP: ${OpenMP_CXX_VERSION}")
