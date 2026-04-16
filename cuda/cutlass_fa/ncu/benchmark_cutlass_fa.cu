#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include "cutlass_fa.h"

int main() {
    std::cout << "CUTLASS Flash Attention Performance Benchmark" << std::endl;
    
    // 测试不同尺寸的性能
    std::vector<std::tuple<int, int, int, int>> test_sizes = {
        {1, 512, 12, 64},    // 小型
        {2, 512, 12, 64},    // 中型
        {1, 1024, 12, 64},   // 大型
        {2, 1024, 16, 64},   // 超大型
    };
    
    std::cout << "Batch x SeqLen x Heads x HeadDim\tAvg Time (μs)\tMemory (MB)" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;
    
    for (auto& [batch_size, seq_len, num_heads, head_dim] : test_sizes) {
        int total_elements = batch_size * seq_len * num_heads * head_dim;
        int lse_size = batch_size * num_heads * seq_len;
        float mem_mb = (4.0f * total_elements * 4 /*4 bytes per float*/ + lse_size * 4) / (1024.0f * 1024.0f);
        
        // 分配设备内存
        float *d_Q, *d_K, *d_V, *d_O, *d_softmax_lse;
        cudaMalloc(&d_Q, total_elements * sizeof(float));
        cudaMalloc(&d_K, total_elements * sizeof(float));
        cudaMalloc(&d_V, total_elements * sizeof(float));
        cudaMalloc(&d_O, total_elements * sizeof(float));
        cudaMalloc(&d_softmax_lse, lse_size * sizeof(float));
        
        // 初始化数据
        std::vector<float> temp(total_elements, 0.1f);
        cudaMemcpy(d_Q, temp.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, temp.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, temp.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
        
        // 预热
        cutlass_flash_attention_forward(
            d_Q, d_K, d_V, d_O, d_softmax_lse,
            batch_size, seq_len, num_heads, head_dim, 1.0f / sqrtf(head_dim)
        );
        cudaDeviceSynchronize();
        
        // 性能测试 - 多次运行取平均值
        const int num_runs = 5;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; ++i) {
            cutlass_flash_attention_forward(
                d_Q, d_K, d_V, d_O, d_softmax_lse,
                batch_size, seq_len, num_heads, head_dim, 1.0f / sqrtf(head_dim)
            );
        }
        cudaDeviceSynchronize();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        auto avg_duration = total_duration.count() / num_runs;
        
        std::cout << std::setw(8) << batch_size << "x" << seq_len 
                  << "x" << num_heads << "x" << head_dim 
                  << "\t\t" << std::setw(8) << avg_duration 
                  << "\t\t" << std::fixed << std::setprecision(2) << mem_mb << std::endl;
        
        // 释放内存
        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_O);
        cudaFree(d_softmax_lse);
    }
    
    std::cout << "Benchmark completed." << std::endl;
    
    return 0;
}