#include "../kernel/rmsnorm_kernel.h"  // 包含API接口定义
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <random>
#include <chrono>

// 辅助函数：初始化随机数据
void initialize_random_data(float* data, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}

int main() {
    // 设置参数
    const int batch_size = 64;
    const int seq_len = 512 ;
    const int hidden_size = 768;
    const int n_rows = batch_size * seq_len; 
    const int n_cols = hidden_size;     
    const float eps = 1e-5f;
    
    std::cout << "Testing RMSNorm CUDA kernels with pure C++ interface..." << std::endl;
    std::cout << "Configuration: " << n_rows << " x " << n_cols << std::endl;
    
    // 分配主机内存
    size_t input_size = n_rows * n_cols * sizeof(float);
    size_t weight_size = n_cols * sizeof(float);
    
    float *h_input = new float[n_rows * n_cols];
    float *h_weight = new float[n_cols];
    float *h_output = new float[n_rows * n_cols];
    float *h_grad_output = new float[n_rows * n_cols];
    float *h_grad_input = new float[n_rows * n_cols];
    float *h_grad_weight = new float[n_cols];
    
    // 初始化数据
    initialize_random_data(h_input, n_rows * n_cols);
    initialize_random_data(h_weight, n_cols);
    initialize_random_data(h_grad_output, n_rows * n_cols);
    
    // 分配设备内存
    float *d_input, *d_weight, *d_output, *d_grad_output, *d_grad_input, *d_grad_weight;
    
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_weight, weight_size);
    cudaMalloc(&d_output, input_size);
    cudaMalloc(&d_grad_output, input_size);
    cudaMalloc(&d_grad_input, input_size);
    cudaMalloc(&d_grad_weight, weight_size);
    
    // 复制数据到设备
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, weight_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output, h_grad_output, input_size, cudaMemcpyHostToDevice);
    
    // 执行前向传播
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    rmsnorm_forward_float(d_input, d_weight, d_output, eps, n_rows, n_cols);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Forward pass completed in " << duration.count() << " us" << std::endl;

    // 执行反向传播
    cudaDeviceSynchronize();
    start = std::chrono::high_resolution_clock::now();
    rmsnorm_backward_float(d_grad_output, d_input, d_weight, d_grad_input, d_grad_weight, 
                           eps, n_rows, n_cols);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Backward pass completed in " << duration.count() << " us" << std::endl;
    
    // 复制结果回主机
    cudaMemcpy(h_output, d_output, input_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_input, d_grad_input, input_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_weight, d_grad_weight, weight_size, cudaMemcpyDeviceToHost);
    
    
    // 清理内存
    delete[] h_input;
    delete[] h_weight;
    delete[] h_output;
    delete[] h_grad_output;
    delete[] h_grad_input;
    delete[] h_grad_weight;
    
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    cudaFree(d_grad_weight);
    
    std::cout << "All operations completed successfully!" << std::endl;
    
    return 0;
}
