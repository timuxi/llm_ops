#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include "cutlass_gemm.h"

// 辅助函数：获取类型名称
template<typename T>
std::string getTypeName();

template<>
std::string getTypeName<float>() {
    return "float";
}

template<>
std::string getTypeName<cutlass::half_t>() {
    return "cutlass::half_t";
}

// 初始化矩阵
template<typename T>
void initializeMatrix(std::vector<T>& matrix, int rows, int cols, int seed = 0) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<T>(sin(static_cast<float>(i + seed)));
    }
}

// CPU参考实现
void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K, 
              float alpha = 1.0f, float beta = 0.0f) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m + k * M] * B[k + n * K]; // Column major
            }
            C[m + n * M] = alpha * sum + beta * C[m + n * M];
        }
    }
}

// 验证结果
template<typename T>
bool verifyResult(const T* computed, const T* reference, int size, float tolerance = 1e-3f) {
    for (int i = 0; i < size; ++i) {
        float diff = fabsf(static_cast<float>(computed[i]) - static_cast<float>(reference[i]));
        if (diff > tolerance) {
            std::cout << "Mismatch at index " << i << ": computed=" 
                      << static_cast<float>(computed[i]) << ", reference=" 
                      << static_cast<float>(reference[i]) << std::endl;
            return false;
        }
    }
    return true;
}

// 测试函数
template<typename T>
bool test_cutlass_gemm(int M = 128, int N = 128, int K = 128) {
    std::cout << "Testing GEMM: " << M << "x" << N << "x" << K 
              << " with " << getTypeName<T>() << std::endl;

    // 分配主机内存
    std::vector<T> h_A(M * K);
    std::vector<T> h_B(K * N);
    std::vector<T> h_C(M * N);
    std::vector<T> h_C_reference(M * N);

    // 初始化矩阵
    initializeMatrix(h_A, M, K, 0);
    initializeMatrix(h_B, K, N, 1);
    initializeMatrix(h_C, M, N, 2);
    h_C_reference = h_C; // 保存原始值用于参考

    // 分配设备内存
    T *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(T));
    cudaMalloc(&d_B, K * N * sizeof(T));
    cudaMalloc(&d_C, M * N * sizeof(T));

    // 复制数据到设备
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), M * N * sizeof(T), cudaMemcpyHostToDevice);

    // 启动GEMM
    auto start = std::chrono::high_resolution_clock::now();
    cudaError_t err = cutlass_gemm(d_A, d_B, d_C, M, N, K, T(1), T(0));
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // 复制结果回主机
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(T), cudaMemcpyDeviceToHost);

    // 如果是浮点类型，计算CPU参考结果进行验证
    if constexpr (std::is_same_v<T, float>) {
        cpu_gemm(h_A.data(), h_B.data(), h_C_reference.data(), M, N, K, 1.0f, 0.0f);
        
        // 验证结果
        bool passed = verifyResult(h_C.data(), h_C_reference.data(), M * N);
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "  " << (passed ? "PASSED" : "FAILED") 
                  << " in " << duration.count() << " μs" << std::endl;
        
        // 计算GFLOPS
        double gflops = (2.0 * M * N * K) / (duration.count() * 1e3);
        std::cout << "  Performance: " << std::fixed << std::setprecision(2) 
                  << gflops << " GFLOPS" << std::endl;
        
        // 清理内存
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        
        return passed;
    } else {
        // 对于half类型，我们主要验证程序是否能运行
        std::cout << "  Executed successfully" << std::endl;
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "  Time: " << duration.count() << " μs" << std::endl;
        
        // 计算理论峰值性能
        double gflops = (2.0 * M * N * K) / (duration.count() * 1e3);
        std::cout << "  Performance: " << std::fixed << std::setprecision(2) 
                  << gflops << " GFLOPS" << std::endl;
        
        // 清理内存
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        
        return true;
    }
}

int main() {
    std::cout << "CUTLASS GEMM Test Suite" << std::endl;
    std::cout << "=======================" << std::endl;

    // 测试不同的矩阵尺寸
    std::vector<std::tuple<int, int, int>> test_sizes = {
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {5 * 1024, 5 * 1024, 128} // 大矩阵测试
    };

    std::cout << "\nTesting with float:" << std::endl;
    for (auto [M, N, K] : test_sizes) {
        if (!test_cutlass_gemm<float>(M, N, K)) {
            std::cout << "Float test failed!" << std::endl;
        }
    }

    std::cout << "\nTesting with half precision:" << std::endl;
    for (auto [M, N, K] : test_sizes) {
        if (!test_cutlass_gemm<cutlass::half_t>(M, N, K)) {
            std::cout << "Half test failed!" << std::endl;
        }
    }

    std::cout << "\nAll tests completed." << std::endl;

    return 0;
}