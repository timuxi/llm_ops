#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <memory>
#include "rmsnorm_kernel.h"


__device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int block_num = blockDim.x / 32;
    int block_idx = threadIdx.x / 32;
    int block_inner_idx = threadIdx.x % 32;

    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (block_inner_idx == 0) shared[block_idx] = val;

    __syncthreads();

    val = (threadIdx.x < block_num) ? shared[threadIdx.x] : 0.0f;

    if (block_idx == 0){
        for(int offset = 16; offset > 0; offset /= 2){
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }

    return val;
}


template<typename INPUT_TYPE, int ILP>
__global__ void rmsnorm_forward_kernel(
    const INPUT_TYPE* __restrict__ input,
    const INPUT_TYPE* __restrict__ weight,
    INPUT_TYPE* __restrict__ output,
    const float eps,
    const int n_rows,
    const int n_cols
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= n_rows) return;

    // Shared memory for reduction
    __shared__ float shared_sum;

    // Compute sum of squares for this row
    float local_sum = 0.0f;
    const float4* fp4_vec = reinterpret_cast<const float4*>(input + row * n_cols);
    INPUT_TYPE v[ILP];
    float4* value = reinterpret_cast<float4*>(&v);
    *value = fp4_vec[tid];
    for(int j = 0; j < ILP; j++){
        float x = static_cast<float>(v[j]);
        local_sum += x * x;
    }

    // Reduction to compute variance
    __syncthreads();
    local_sum = blockReduceSum(local_sum);
    
    if (tid == 0) {
        shared_sum = rsqrtf(local_sum / n_cols + eps);
    }
    __syncthreads();
    const float inv_rms = shared_sum;
    INPUT_TYPE w[ILP];
    float4* weight_value = reinterpret_cast<float4*>(&w);
    float4* out_fp4_vec = reinterpret_cast<float4*>(output + row * n_cols);
    const float4* weight_fp4_vec = reinterpret_cast<const float4*>(weight);

    *weight_value = weight_fp4_vec[tid];
    for(int j = 0;j < ILP; j++){
        float x = static_cast<float>(v[j]) * inv_rms * static_cast<float>(w[j]);
        v[j] = static_cast<INPUT_TYPE>(x);
    }
    out_fp4_vec[tid] = *value; 
}

template<typename INPUT_TYPE>
__global__ void rmsnorm_backward_kernel(
    const INPUT_TYPE* __restrict__ grad_output,
    const INPUT_TYPE* __restrict__ input,
    const INPUT_TYPE* __restrict__ weight,
    INPUT_TYPE* __restrict__ grad_input,
    INPUT_TYPE* __restrict__ grad_weight,
    const float eps,
    const int n_rows,
    const int n_cols
) {
    // Process grad_weight first (reduction across rows for each column)
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int col = blockIdx.x;  // Each block handles one column of weight
    
    if (col >= n_cols) return;
    
    // Calculate partial sum for this column across all rows this block handles
    float local_grad_weight_sum = 0.0f;
    for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < n_rows; row += gridDim.y * blockDim.y) {
        if (row >= n_rows) break;
        
        const int idx = row * n_cols + col;
        float input_val = static_cast<float>(input[idx]);
        float grad_out_val = static_cast<float>(grad_output[idx]);
        
        // Compute variance for this row
        float row_var = 0.0f;
        for (int c = tid; c < n_cols; c += block_size) {
            const int row_idx = row * n_cols + c;
            float row_input_val = static_cast<float>(input[row_idx]);
            row_var += row_input_val * row_input_val;
        }
        
        __shared__ float shared_row_var;
        row_var = blockReduceSum(row_var);
        if (tid == 0) {
            shared_row_var = rsqrtf(row_var / n_cols + eps);
        }
        __syncthreads();
        
        float inv_rms = shared_row_var;
        float norm_val = input_val * inv_rms;
        local_grad_weight_sum += grad_out_val * norm_val;
    }
    
    local_grad_weight_sum = blockReduceSum(local_grad_weight_sum);
    if (tid == 0) {
        atomicAdd(&((float*)grad_weight)[col], local_grad_weight_sum);
    }
    
    // Process grad_input for this thread block
    for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < n_rows; row += gridDim.y * blockDim.y) {
        if (row >= n_rows) break;
        
        // Recompute variance for this row
        float row_var = 0.0f;
        for (int c = tid; c < n_cols; c += block_size) {
            const int row_idx = row * n_cols + c;
            float row_input_val = static_cast<float>(input[row_idx]);
            row_var += row_input_val * row_input_val;
        }
        
        __shared__ float shared_row_var2;
        row_var = blockReduceSum(row_var);
        if (tid == 0) {
            shared_row_var2 = rsqrtf(row_var / n_cols + eps);
        }
        __syncthreads();
        
        float inv_rms = shared_row_var2;
        
        const int idx = row * n_cols + col;
        float input_val = static_cast<float>(input[idx]);
        float grad_out_val = static_cast<float>(grad_output[idx]);
        float weight_val = static_cast<float>(weight[col]);
        
        // Compute grad_input
        float g_o_w = grad_out_val * weight_val;
        float mean_square = row_var / n_cols;
        float deno = powf(mean_square + eps, 1.5f);
        float grad_input_val = g_o_w * inv_rms - 
                              g_o_w * input_val * input_val / (n_cols * deno);
        
        grad_input[idx] = static_cast<INPUT_TYPE>(grad_input_val);
    }
}

// C++接口函数
extern "C" {
    void rmsnorm_forward_float(const float* input, const float* weight, float* output, 
                            float eps, int n_rows, int n_cols) {
        int threads = std::min(1024, n_cols / 4);
        int blocks = n_rows;

        rmsnorm_forward_kernel<float, 4><<<blocks, threads>>>(
            input, weight, output, eps, n_rows, n_cols
        );
    }

    void rmsnorm_backward_float(const float* grad_output, const float* input, const float* weight, float* grad_input, float* grad_weight, 
                                float eps, int n_rows, int n_cols) {
        // Use 2D grid for better handling of grad_weight reduction
        const int threads_x = std::min(32, n_cols);  // Threads for cols
        const int threads_y = std::min(32, n_rows / 4 + 1);  // Threads for rows
        dim3 block_size(threads_x, threads_y);
        
        const int blocks_x = (n_cols + threads_x - 1) / threads_x;  // One block per col chunk
        const int blocks_y = (n_rows + threads_y * 4 - 1) / (threads_y * 4);  // Multiple rows per block
        dim3 grid_size(blocks_x, blocks_y);
        
        rmsnorm_backward_kernel<float><<<grid_size, block_size>>>(
            grad_output, input, weight, grad_input, grad_weight, eps, n_rows, n_cols
        );
    }
}