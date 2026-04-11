#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cutlass_gemm.h"

// 使用CUTLASS实现的SGEMM（单精度GEMM）
cudaError_t cutlass_sgemm(
    const float* d_A,  // M x K 矩阵
    const float* d_B,  // K x N 矩阵
    float* d_C,        // M x N 矩阵
    int M, int N, int K,
    float alpha, float beta) {
    
    // 定义CUTLASS GEMM操作符
    using Sgemm = cutlass::gemm::device::Gemm<
        float,                          // 输入A的数据类型
        cutlass::layout::ColumnMajor,   // A的布局
        float,                          // 输入B的数据类型
        cutlass::layout::ColumnMajor,   // B的布局
        float,                          // 输出C的数据类型
        cutlass::layout::ColumnMajor>;  // C的布局

    // 创建GEMM参数
    typename Sgemm::Arguments arguments{
        {M, N, K},                    // {m, n, k} 矩阵维度
        {d_A, M},                     // A矩阵及其ldA（leading dimension）
        {d_B, K},                     // B矩阵及其ldB
        {d_C, M},                     // C矩阵及其ldC（输入，用于beta系数）
        {d_C, M},                     // 输出D矩阵及其ldD
        {alpha, beta}                 // {alpha, beta} 缩放系数
    };

    // 创建GEMM实例
    Sgemm gemm_op;

    // 检查参数是否有效
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS SGEMM不能实现指定的参数!" << std::endl;
        return cudaErrorInvalidValue;
    }

    // 初始化GPU上的矩阵
    int workspace_size = gemm_op.get_workspace_size(arguments);
    void* workspace_ptr = nullptr;
    if (workspace_size) {
        cudaMalloc(&workspace_ptr, workspace_size);
    }

    // 执行GEMM操作 C = alpha * A * B + beta * C
    status = gemm_op(arguments, workspace_ptr);
    
    if (workspace_ptr) {
        cudaFree(workspace_ptr);
    }

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS SGEMM执行失败!" << std::endl;
        return cudaGetLastError(); // 返回最近的CUDA错误
    }

    return cudaSuccess;
}

// 使用CUTLASS实现的HGEMM（半精度GEMM）
cudaError_t cutlass_hgemm(
    const cutlass::half_t* d_A,  // M x K 矩阵
    const cutlass::half_t* d_B,  // K x N 矩阵
    cutlass::half_t* d_C,        // M x N 矩阵
    int M, int N, int K,
    float alpha, float beta) {
    
    // 定义CUTLASS HGEMM操作符
    using Hgemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,              // 输入A的数据类型
        cutlass::layout::ColumnMajor, // A的布局
        cutlass::half_t,              // 输入B的数据类型
        cutlass::layout::ColumnMajor, // B的布局
        cutlass::half_t,              // 输出C的数据类型
        cutlass::layout::ColumnMajor>;// C的布局

    // 创建GEMM参数
    typename Hgemm::Arguments arguments{
        {M, N, K},                           // {m, n, k} 矩阵维度
        {d_A, M},                            // A矩阵及其ldA
        {d_B, K},                            // B矩阵及其ldB
        {d_C, M},                            // C矩阵及其ldC（输入，用于beta系数）
        {d_C, M},                            // 输出D矩阵及其ldD
        {cutlass::half_t(alpha), cutlass::half_t(beta)}  // {alpha, beta} 缩放系数
    };

    // 创建GEMM实例
    Hgemm gemm_op;

    // 检查参数是否有效
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS HGEMM不能实现指定的参数!" << std::endl;
        return cudaErrorInvalidValue;
    }

    // 初始化GPU上的矩阵
    int workspace_size = gemm_op.get_workspace_size(arguments);
    void* workspace_ptr = nullptr;
    if (workspace_size) {
        cudaMalloc(&workspace_ptr, workspace_size);
    }

    // 执行GEMM操作
    status = gemm_op(arguments, workspace_ptr);
    
    if (workspace_ptr) {
        cudaFree(workspace_ptr);
    }

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS HGEMM执行失败!" << std::endl;
        return cudaGetLastError(); // 返回最近的CUDA错误
    }

    return cudaSuccess;
}

// 模板化的GEMM函数实现
template<typename Element>
cudaError_t cutlass_gemm(
    const Element* d_A,
    const Element* d_B,
    Element* d_C,
    int M, int N, int K,
    Element alpha, Element beta) {
    
    // 根据Element类型选择适当的GEMM实现
    if constexpr (std::is_same_v<Element, float>) {
        return cutlass_sgemm(d_A, d_B, d_C, M, N, K, 
                             static_cast<float>(alpha), static_cast<float>(beta));
    } else if constexpr (std::is_same_v<Element, cutlass::half_t>) {
        return cutlass_hgemm(reinterpret_cast<const cutlass::half_t*>(d_A),
                             reinterpret_cast<const cutlass::half_t*>(d_B),
                             reinterpret_cast<cutlass::half_t*>(d_C),
                             M, N, K,
                             static_cast<float>(alpha), static_cast<float>(beta));
    } else {
        std::cerr << "不支持的数据类型!" << std::endl;
        return cudaErrorInvalidValue;
    }
}

// 显式模板实例化
template cudaError_t cutlass_gemm<float>(const float*, const float*, float*, int, int, int, float, float);
template cudaError_t cutlass_gemm<cutlass::half_t>(const cutlass::half_t*, const cutlass::half_t*, cutlass::half_t*, int, int, int, cutlass::half_t, cutlass::half_t);