cmake_minimum_required(VERSION 3.18)
project(cutlass_add LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDA REQUIRED)

# 设置CUTLASS路径
set(CUTLASS_PATH "/home/xcy/cutlass")

# 包含CUTLASS头文件
include_directories(${CUTLASS_PATH}/include)

# 设置CUDA架构（根据您的GPU调整）
set(CMAKE_CUDA_ARCHITECTURES 75)  # For Turing architecture (RTX 20xx series)
# set(CMAKE_CUDA_ARCHITECTURES 80)  # For Ampere architecture (RTX 30xx series)
# set(CMAKE_CUDA_ARCHITECTURES 86)  # For RTX 30xx series

# 添加可执行文件
add_executable(test_cutlass_add 
    test_cutlass_add.cu
    cutlass_add_impl.cu
    cutlass_tensor_add.cu
    cutlass_gemm_add.cu
)

# 设置CUDA编译选项
set_property(TARGET test_cutlass_add PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# 链接库
target_link_libraries(test_cutlass_add ${CUDA_LIBRARIES})

# CUDA编译器选项
target_compile_options(test_cutlass_add PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-arch=sm_75>
    $<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:SHELL:--use_fast_math>
)