cmake_minimum_required(VERSION 3.18)
project(cutlass_fa LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 设置CUTLASS路径
set(CUTLASS_PATH "/home/xcy/Block-Sparse-Attention/csrc/cutlass")

# 查找CUDA
find_package(CUDA REQUIRED)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES 75)

# 添加头文件路径
include_directories(${CUTLASS_PATH}/include)
include_directories(${PROJECT_SOURCE_DIR}/../interface)

# 定义库
add_library(cutlass_fa
    ${PROJECT_SOURCE_DIR}/../kernel/cutlass_fa.cu
)

set_target_properties(cutlass_fa PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
)

# 定义测试可执行文件
add_executable(test_cutlass_fa
    ${PROJECT_SOURCE_DIR}/../ncu/test_cutlass_fa.cu
)

target_link_libraries(test_cutlass_fa cutlass_fa)

# 定义基准测试可执行文件
add_executable(benchmark_cutlass_fa
    ${PROJECT_SOURCE_DIR}/../ncu/benchmark_cutlass_fa.cu
)

target_link_libraries(benchmark_cutlass_fa cutlass_fa)