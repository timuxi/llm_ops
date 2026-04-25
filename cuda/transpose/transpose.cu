#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <memory>



void cpu_transpose(float* input, float* out, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            out[j * M + i] = input[i * N + j];
        }
    }
}

bool verify_result(float* gpu_result, float* cpu_result, int size, float eps = 0.01f) {
    for (int i = 0; i < size; ++i) {
        if (fabs(gpu_result[i] - cpu_result[i]) / cpu_result[i] > eps) {
            printf("验证失败: GPU结果 %f vs CPU结果 %f\n", gpu_result[i], cpu_result[i]);
            return false;
        }
    }
    printf("验证成功\n");
    return true;
}


__global__ void transpose_kernel(
    float* input,
    float* output,
    int M,
    int N
){
    __shared__ float smem[32][33];
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int block_size_x = blockDim.x;
    int block_size_y = blockDim.y;

    int inner_row_idx = threadIdx.y;
    int inner_col_idx = threadIdx.x;
    int abs_row_idx = block_y * block_size_y + inner_row_idx;
    int abs_col_idx = block_x * block_size_x + inner_col_idx;
    int abs_input_idx = abs_row_idx * N + abs_col_idx;
    int abs_out_idx = abs_col_idx * N + abs_row_idx;

    if(abs_input_idx >= M * N) {
        return;
    }

    smem[inner_row_idx][inner_col_idx] = input[abs_input_idx];
    __syncthreads();
    output[abs_out_idx] = smem[inner_row_idx][inner_col_idx];
}


void transpose(float* d_input, float* d_output, int M, int N) {
    dim3 block_size(32, 32);
    dim3 grid_size((M + 32 - 1) / 32, (N + 32 - 1) / 32);

    transpose_kernel<<<grid_size, block_size>>>(
        d_input, d_output, M, N
    );
}


int main() {
    // 测试数据大小
    int M = 64;
    int N = 64;
    size_t size = M * N * sizeof(float);
    
    printf("开始CUDA transpose测试，数据大小: %d 个元素\n", M * N);
    
    // 分配主机内存并初始化数据
    float *h_input = (float*)malloc(size);
    float *h_output_cpu = (float*)malloc(size);
    float* h_output_gpu = (float*)malloc(size);

    // 初始化数据：简单的递增序列
    for (int i = 0; i < M * N; ++i) {
        h_input[i] = (float)(i) / 10.0f;
    }
    cpu_transpose(h_input, h_output_cpu, M, N);
    // 分配设备内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    // 复制数据到设备
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // GPU计算
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // 执行reduce
    transpose(d_input, d_output, M, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_duration;
    cudaEventElapsedTime(&gpu_duration, start, stop);
    
    // 复制结果回主机
    cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost);
    
    printf("GPU计算完成，耗时: %.3f ms\n", gpu_duration);

    // 验证结果
    bool success = verify_result(h_output_gpu, h_output_cpu, M * N);
    
    // 清理资源
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    if (success) {
        printf("测试通过!\n");
        return 0;
    } else {
        printf("测试失败!\n");
        return -1;
    }
}