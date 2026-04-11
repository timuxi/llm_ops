#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cuda_runtime.h>
#include <iostream>
#include <type_traits>
#include "cutlass_gemm.h"

// 通用模板函数，根据数据类型选择合适的CUTLASS GEMM实现
template<typename Element>
cudaError_t cutlass_gemm_template(
    const Element* d_A,
    const Element* d_B,
    Element* d_C,
    int M, int N, int K,
    Element alpha = Element(1), Element beta = Element(0)) {
    
    // 这是一个通用实现，针对不同数据类型会有特定的特化
    std::cerr << "未定义此数据类型的GEMM实现!" << std::endl;
    return cudaErrorInvalidValue;
}

// float类型的特化
template<>
cudaError_t cutlass_gemm_template<float>(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M, int N, int K,
    float alpha, float beta) {
    
    using ColumnMajor = cutlass::layout::ColumnMajor;
    
    // 定义CUTLASS GEMM操作符 - 单精度
    using Gemm = cutlass::gemm::device::Gemm<
        float, ColumnMajor,    // A的数据类型和布局
        float, ColumnMajor,    // B的数据类型和布局
        float, ColumnMajor>;   // C/D的数据类型和布局

    // 准备GEMM参数
    typename Gemm::Arguments arguments{
        {M, N, K},           // {m, n, k} - 矩阵维度
        {d_A, M},            // A矩阵指针和leading dimension
        {d_B, K},            // B矩阵指针和leading dimension
        {d_C, M},            // C矩阵指针和leading dimension (输入)
        {d_C, M},            // D矩阵指针和leading dimension (输出)
        {alpha, beta}        // 缩放因子
    };

    // 创建GEMM操作实例
    Gemm gemm_op;

    // 验证参数
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM无法实现指定参数!" << std::endl;
        return cudaErrorInvalidValue;
    }

    // 获取所需的工作空间大小并分配
    size_t workspace_size = gemm_op.get_workspace_size(arguments);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
        if (!workspace) {
            std::cerr << "无法分配工作空间!" << std::endl;
            return cudaErrorMemoryAllocation;
        }
    }

    // 执行GEMM操作: D = alpha * A * B + beta * C
    status = gemm_op(arguments, workspace);

    // 释放工作空间
    if (workspace) {
        cudaFree(workspace);
    }

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM执行失败!" << std::endl;
        return cudaGetLastError(); // 使用正确的错误返回
    }

    return cudaSuccess;
}

// half_t类型的特化
template<>
cudaError_t cutlass_gemm_template<cutlass::half_t>(
    const cutlass::half_t* d_A,
    const cutlass::half_t* d_B,
    cutlass::half_t* d_C,
    int M, int N, int K,
    cutlass::half_t alpha, cutlass::half_t beta) {
    
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using Half = cutlass::half_t;
    
    // 定义CUTLASS GEMM操作符 - 半精度
    using Gemm = cutlass::gemm::device::Gemm<
        Half, ColumnMajor,     // A的数据类型和布局
        Half, ColumnMajor,     // B的数据类型和布局
        Half, ColumnMajor>;    // C/D的数据类型和布局

    // 准备GEMM参数
    typename Gemm::Arguments arguments{
        {M, N, K},                       // {m, n, k} - 矩阵维度
        {d_A, M},                        // A矩阵指针和leading dimension
        {d_B, K},                        // B矩阵指针和leading dimension
        {d_C, M},                        // C矩阵指针和leading dimension (输入)
        {d_C, M},                        // D矩阵指针和leading dimension (输出)
        {alpha, beta}                    // 缩放因子
    };

    // 创建GEMM操作实例
    Gemm gemm_op;

    // 验证参数
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS HGEMM无法实现指定参数!" << std::endl;
        return cudaErrorInvalidValue;
    }

    // 获取所需的工作空间大小并分配
    size_t workspace_size = gemm_op.get_workspace_size(arguments);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
        if (!workspace) {
            std::cerr << "无法分配工作空间!" << std::endl;
            return cudaErrorMemoryAllocation;
        }
    }

    // 执行GEMM操作: D = alpha * A * B + beta * C
    status = gemm_op(arguments, workspace);

    // 释放工作空间
    if (workspace) {
        cudaFree(workspace);
    }

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS HGEMM执行失败!" << std::endl;
        return cudaGetLastError(); // 使用正确的错误返回
    }

    return cudaSuccess;
}

// 重定向到模板实现
template<typename Element>
cudaError_t cutlass_gemm(
    const Element* d_A,
    const Element* d_B,
    Element* d_C,
    int M, int N, int K,
    Element alpha, Element beta) {
    return cutlass_gemm_template(d_A, d_B, d_C, M, N, K, alpha, beta);
}

// 显式模板实例化
template cudaError_t cutlass_gemm<float>(const float*, const float*, float*, int, int, int, float, float);
template cudaError_t cutlass_gemm<cutlass::half_t>(const cutlass::half_t*, const cutlass::half_t*, cutlass::half_t*, int, int, int, cutlass::half_t, cutlass::half_t);