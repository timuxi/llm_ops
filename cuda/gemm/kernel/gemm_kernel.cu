#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <memory>
#include "gemm_kernel.h"

#define TILE_SIZE 16

__device__ int AlignUp(int x, int align) {
    return (x + align - 1) / align * align;
}

__device__ int CeilDiv(int x, int align) {
    return AlignUp(x, align) / align;
}

template<typename INPUT_TYPE>
__global__ void gemm_kernel(
    const INPUT_TYPE* __restrict__ A,
    const INPUT_TYPE* __restrict__ B,
    INPUT_TYPE* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float A_tile[2][TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[2][TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }
    float sum = 0.0f;
    
    for (int k_idx = 0; k_idx < CeilDiv(K, TILE_SIZE); ++k_idx) {
        int ping_pong_idx = k_idx % 2;
        A_tile[ping_pong_idx][threadIdx.y][threadIdx.x] = A[row * K + k_idx * TILE_SIZE + threadIdx.x];
        B_tile[ping_pong_idx][threadIdx.y][threadIdx.x] = B[(k_idx * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += A_tile[ping_pong_idx][threadIdx.y][k] * B_tile[ping_pong_idx][k][threadIdx.x];
        }
    }

    // 将结果写回全局内存
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
