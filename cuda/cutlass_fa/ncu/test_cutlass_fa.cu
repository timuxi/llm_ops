#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <random>
#include <cutlass/half.h>
#include "cutlass_fa.h"

// 辅助函数：初始化CUDA数组为随机值
void initialize_random(cutlass::half_t* data, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    std::vector<__half> host_data(size);
    for (int i = 0; i < size; ++i) {
        host_data[i] = __float2half(dis(gen));  // Convert float to half
    }
    
    cudaMemcpy(data, host_data.data(), size * sizeof(__half), cudaMemcpyHostToDevice);
}

// 辅助函数：比较两个CUDA数组
bool compare_arrays(const cutlass::half_t* a, const cutlass::half_t* b, int size, float tolerance = 1e-3f) {
    std::vector<__half> host_a(size), host_b(size);
    cudaMemcpy(host_a.data(), a, size * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_b.data(), b, size * sizeof(__half), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < size; ++i) {
        if (abs(__half2float(host_a[i]) - __half2float(host_b[i])) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << __half2float(host_a[i]) << " vs " << __half2float(host_b[i]) << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    std::cout << "CUTLASS Flash Attention Test Suite" << std::endl;
    
    // 测试参数
    int batch_size = 2;
    int seq_len = 512;
    int num_heads = 8;
    int head_dim = 64;
    float scale_factor = 1.0f / sqrtf(head_dim);
    
    std::cout << "Testing with: " << batch_size << "x" << seq_len << "x" 
              << num_heads << "x" << head_dim << std::endl;
    
    // 计算总元素数
    int total_elements = batch_size * seq_len * num_heads * head_dim;
    int lse_size = batch_size * num_heads * seq_len;
    
    // 分配设备内存
    cutlass::half_t *d_Q, *d_K, *d_V, *d_O;
    float *d_softmax_lse;
    cudaMalloc(&d_Q, total_elements * sizeof(cutlass::half_t));
    cudaMalloc(&d_K, total_elements * sizeof(cutlass::half_t));
    cudaMalloc(&d_V, total_elements * sizeof(cutlass::half_t));
    cudaMalloc(&d_O, total_elements * sizeof(cutlass::half_t));
    cudaMalloc(&d_softmax_lse, lse_size * sizeof(float));
    
    // 初始化数据
    initialize_random(d_Q, total_elements);
    initialize_random(d_K, total_elements);
    initialize_random(d_V, total_elements);
    
    // 执行Flash Attention前向传播
    auto start_time = std::chrono::high_resolution_clock::now();
    cudaError_t forward_result = cutlass_flash_attention_forward(
        d_Q, d_K, d_V, d_O, d_softmax_lse,
        batch_size, seq_len, num_heads, head_dim, scale_factor
    );
    auto end_time = std::chrono::high_resolution_clock::now();
    
    if (forward_result != cudaSuccess) {
        std::cout << "Forward pass failed: " << cudaGetErrorString(forward_result) << std::endl;
        return -1;
    }
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "  Forward PASS in " << duration.count() << " μs" << std::endl;
    
    // 验证输出不为零（至少有些计算发生了）
    std::vector<__half> output_check(10);
    cudaMemcpy(output_check.data(), d_O, 10 * sizeof(__half), cudaMemcpyDeviceToHost);
    
    bool output_non_zero = false;
    for (int i = 0; i < 10; ++i) {
        if (abs(__half2float(output_check[i])) > 1e-6) {
            output_non_zero = true;
            break;
        }
    }
    
    if (output_non_zero) {
        std::cout << "  Output verification: PASSED" << std::endl;
    } else {
        std::cout << "  Output verification: FAILED - all outputs are zero" << std::endl;
        return -1;
    }
    
    // 测试较小的尺寸以加快测试速度
    int small_seq_len = 64;
    int small_total = batch_size * small_seq_len * num_heads * head_dim;
    int small_lse = batch_size * num_heads * small_seq_len;
    
    cutlass::half_t *d_small_Q, *d_small_K, *d_small_V, *d_small_O;
    float *d_small_softmax_lse;
    cudaMalloc(&d_small_Q, small_total * sizeof(cutlass::half_t));
    cudaMalloc(&d_small_K, small_total * sizeof(cutlass::half_t));
    cudaMalloc(&d_small_V, small_total * sizeof(cutlass::half_t));
    cudaMalloc(&d_small_O, small_total * sizeof(cutlass::half_t));
    cudaMalloc(&d_small_softmax_lse, small_lse * sizeof(float));
    
    initialize_random(d_small_Q, small_total);
    initialize_random(d_small_K, small_total);
    initialize_random(d_small_V, small_total);
    
    // 测试小尺寸的前向传播
    std::cout << "Testing with smaller size: " << batch_size << "x" << small_seq_len << "x" 
              << num_heads << "x" << head_dim << std::endl;
              
    forward_result = cutlass_flash_attention_forward(
        d_small_Q, d_small_K, d_small_V, d_small_O, d_small_softmax_lse,
        batch_size, small_seq_len, num_heads, head_dim, scale_factor
    );
    
    if (forward_result != cudaSuccess) {
        std::cout << "Small forward pass failed: " << cudaGetErrorString(forward_result) << std::endl;
        return -1;
    }
    
    std::cout << "  Small forward PASS" << std::endl;
    
    // 释放内存
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_softmax_lse);
    cudaFree(d_small_Q);
    cudaFree(d_small_K);
    cudaFree(d_small_V);
    cudaFree(d_small_O);
    cudaFree(d_small_softmax_lse);
    
    std::cout << "All tests completed." << std::endl;
    
    return 0;
}