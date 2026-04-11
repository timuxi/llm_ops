#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include "cutlass_gemm.h"

// 性能测试函数
template<typename T>
double benchmark_cutlass_gemm(int M, int N, int K, int iterations = 5) {
    // 分配主机内存
    std::vector<T> h_A(M * K);
    std::vector<T> h_B(K * N);
    std::vector<T> h_C(M * N);

    // 初始化矩阵
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<T>(static_cast<float>(i % 100) / 100.0f);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<T>(static_cast<float>((i + 1) % 100) / 100.0f);
    }
    for (int i = 0; i < M * N; ++i) {
        h_C[i] = static_cast<T>(0);
    }

    // 分配设备内存
    T *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(T));
    cudaMalloc(&d_B, K * N * sizeof(T));
    cudaMalloc(&d_C, M * N * sizeof(T));

    // 复制数据到设备
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), M * N * sizeof(T), cudaMemcpyHostToDevice);

    // 预热
    cutlass_gemm(d_A, d_B, d_C, M, N, K, T(1), T(0));
    cudaDeviceSynchronize();

    // 性能测试
    auto total_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        cudaError_t err = cutlass_gemm(d_A, d_B, d_C, M, N, K, T(1), T(0));
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            return -1.0;
        }
    }
    auto total_end = std::chrono::high_resolution_clock::now();

    // 计算平均时间
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    double avg_time = static_cast<double>(total_duration.count()) / iterations;

    // 清理内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return avg_time;
}

int main() {
    std::cout << "CUTLASS GEMM Performance Benchmark" << std::endl;
    std::cout << "==================================" << std::endl;

    // 测试不同的矩阵尺寸
    std::vector<std::tuple<int, int, int>> test_sizes = {
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {5 * 1024, 5 * 1024, 128}
    };

    std::cout << "\nFloat Performance:" << std::endl;
    std::cout << "Size (MxNxK)\tAvg Time (μs)\tGFLOPS" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    for (auto [M, N, K] : test_sizes) {
        double time_us = benchmark_cutlass_gemm<float>(M, N, K);
        if (time_us > 0) {
            // 计算GFLOPS：2 * M * N * K / (time_in_seconds * 1e9)
            double gflops = (2.0 * M * N * K) / (time_us * 1e3);
            std::cout << M << "x" << N << "x" << K << "\t\t" 
                      << std::fixed << std::setprecision(2) 
                      << time_us << "\t\t" << gflops << std::endl;
        } else {
            std::cout << M << "x" << N << "x" << K << "\t\tERROR\t\tERROR" << std::endl;
        }
    }

    std::cout << "\nHalf Precision Performance:" << std::endl;
    std::cout << "Size (MxNxK)\tAvg Time (μs)\tGFLOPS" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    for (auto [M, N, K] : test_sizes) {
        // 仅对较小的矩阵测试half精度，因为可能受硬件限制
        if (M <= 512) {  // 限制half精度测试的大小
            double time_us = benchmark_cutlass_gemm<cutlass::half_t>(M, N, K);
            if (time_us > 0) {
                // 计算GFLOPS：2 * M * N * K / (time_in_seconds * 1e9)
                double gflops = (2.0 * M * N * K) / (time_us * 1e3);
                std::cout << M << "x" << N << "x" << K << "\t\t" 
                          << std::fixed << std::setprecision(2) 
                          << time_us << "\t\t" << gflops << std::endl;
            } else {
                std::cout << M << "x" << N << "x" << K << "\t\tERROR\t\tERROR" << std::endl;
            }
        } else {
            std::cout << M << "x" << N << "x" << K << "\t\tSKIP\t\tSKIP" << std::endl;
        }
    }

    std::cout << "\nBenchmark completed." << std::endl;

    return 0;
}