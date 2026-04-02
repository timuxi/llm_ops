#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <memory>
#include "gemm_kernel.h"


template<typename INPUT_TYPE>
__global__ void gemm_kernel(
    const INPUT_TYPE* __restrict__ A,
    const INPUT_TYPE* __restrict__ B,
    INPUT_TYPE* __restrict__ C,
    int M, int N, int K
) {
    // 计算当前线程对应的行和列
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }
    float sum = 0.0f;
    // 内积循环
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    C[row * N + col] = sum;
}

// C++接口函数
extern "C" {
    void gemm_kernel_float(const float* A, const float* B, float* C, int M, int N, int K) {
        dim3 blockDim(16, 16);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                     (M + blockDim.y - 1) / blockDim.y);

        gemm_kernel<float><<<gridDim, blockDim>>>(
            A, B, C, M, N, K
        );
    }
}
