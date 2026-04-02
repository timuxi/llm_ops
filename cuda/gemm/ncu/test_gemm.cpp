#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../kernel/gemm_kernel.h" 
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <random>
#include <chrono>

// 主机端验证函数（简单对比）
bool verify_result(float *h_C, float *h_C_ref, int M, int N, float eps = 0.001f) {
    for (int i = 0; i < M * N; ++i) {
        if (fabs(h_C[i] - h_C_ref[i]) > eps) {
            printf("验证失败 at index %d: %f vs %f\n", i, h_C[i], h_C_ref[i]);
            return false;
        }
    }
    return true;
}


int main() {
    // 矩阵维度设置
    int M = 5 * 1024;   // A的行数，C的行数
    int N = 5 * 1024;   // B的列数，C的列数
    int K = 128;   // A的列数，B的行数

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // 分配主机内存并初始化数据
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_ref = (float*)malloc(size_C); // 用于CPU计算结果对比

    // 简单初始化：A和B的元素设为1.0，这样C的每个元素应为K
    for (int i = 0; i < M * K; ++i) h_A[i] = (float) (i % 33) / 10.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float) (i % 33) / 10.0f;
    // 可选：随机初始化
    // for (int i = 0; i < M*K; ++i) h_A[i] = rand() / (float)RAND_MAX;
    // for (int i = 0; i < K*N; ++i) h_B[i] = rand() / (float)RAND_MAX;

    // CPU端计算参考结果（简单三重循环）
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C_ref[i * N + j] = sum;
        }
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 将数据从主机拷贝到设备
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    // 调用核函数
    gemm_kernel_float(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Forward pass completed in " << duration.count() << " us" << std::endl;


    // 将结果从设备拷贝回主机
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // 验证结果
    bool correct = verify_result(h_C, h_C_ref, M, N);
    if (correct) {
        printf("结果正确！\n");
    } else {
        printf("结果错误！\n");
    }

    // 释放资源
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}