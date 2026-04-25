#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <memory>


float cpu_reduce_sum(float* data, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += data[i];
    }
    return sum;
}

bool verify_result(float gpu_result, float cpu_result, float eps = 0.01f) {
    if (fabs(gpu_result - cpu_result) / cpu_result > eps) {
        printf("验证失败: GPU结果 %f vs CPU结果 %f\n", gpu_result, cpu_result);
        return false;
    }
    printf("验证成功: GPU结果 %f vs CPU结果 %f\n", gpu_result, cpu_result);
    return true;
}



__global__ void reduce_kernel(
    float* input,
    float* output,
    int N
){
    int tid = threadIdx.x;
    int bix = blockIdx.x;
    int block_size = blockDim.x;
    int col = bix * block_size + tid;
    float val = 0.0f;
    if (col < N) {
        val = input[col];
    }

    __shared__ float smem[32];
    int wrap_idx = tid / 32;
    int warp_inner_idx = tid % 32;


    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if(warp_inner_idx == 0){
        smem[wrap_idx] = val;
    }
    __syncthreads();

    if(wrap_idx == 0){
        val = smem[warp_inner_idx];
    } else {
        val = 0.0f;
    }

    __syncthreads();

    for(int offset=16;offset>0;offset/=2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if(tid == 0){
        atomicAdd(&output[0], val);
    }
    
}


void reduce_sum(float* d_input, float* d_output, int N) {
    int thead_size = std::min(1024, N);
    int block_size = (N + thead_size - 1) / thead_size;
    printf("block_size %d, thead_size %d\n", block_size, thead_size);
    reduce_kernel<<<block_size, thead_size>>>(
        d_input, d_output, N
    );
}

int main() {
    // 测试数据大小
    int N = 177 + 1024;
    size_t size = N * sizeof(float);
    
    printf("开始CUDA reduce测试，数据大小: %d 个元素\n", N);
    
    // 分配主机内存并初始化数据
    float *h_input = (float*)malloc(size);
    float h_output_gpu;
    float h_output_cpu;
    
    // 初始化数据：简单的递增序列
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)(i) / 10.0f;
    }
    h_output_cpu = cpu_reduce_sum(h_input, N);
    // 分配设备内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // GPU计算
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // 执行reduce
    reduce_sum(d_input, d_output, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_duration;
    cudaEventElapsedTime(&gpu_duration, start, stop);
    
    // 复制结果回主机
    cudaMemcpy(&h_output_gpu, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("GPU计算完成，耗时: %.3f ms\n", gpu_duration);

    // 验证结果
    bool success = verify_result(h_output_gpu, h_output_cpu);
    
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