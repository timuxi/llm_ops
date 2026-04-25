#ifndef CUTLASS_FA_H
#define CUTLASS_FA_H

#include <cuda_runtime.h>
#include <cutlass/coord.h>
#include <cutlass/half.h>

// FlashAttention前向传播函数声明
cudaError_t cutlass_flash_attention_forward(
    const cutlass::half_t* query,     // 查询矩阵 Q [batch_size, seq_len, num_heads, head_dim]
    const cutlass::half_t* key,       // 键矩阵 K [batch_size, seq_len, num_heads, head_dim] 
    const cutlass::half_t* value,     // 值矩阵 V [batch_size, seq_len, num_heads, head_dim]
    cutlass::half_t* output,          // 输出矩阵 O [batch_size, seq_len, num_heads, head_dim]
    float* softmax_lse,               // Softmax对数求和指数 [batch_size, num_heads, seq_len]
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    float scale_factor = 1.0f
);

// FlashAttention反向传播函数声明
cudaError_t cutlass_flash_attention_backward(
    const cutlass::half_t* grad_output,   // 输出梯度 [batch_size, seq_len, num_heads, head_dim]
    const cutlass::half_t* query,         // 查询矩阵 Q
    const cutlass::half_t* key,           // 键矩阵 K
    const cutlass::half_t* value,         // 值矩阵 V
    const cutlass::half_t* output,        // 前向输出 O
    const float* softmax_lse,             // Softmax对数求和指数
    cutlass::half_t* grad_query,          // 查询梯度 [batch_size, seq_len, num_heads, head_dim]
    cutlass::half_t* grad_key,            // 键梯度 [batch_size, seq_len, num_heads, head_dim]
    cutlass::half_t* grad_value,          // 值梯度 [batch_size, seq_len, num_heads, head_dim]
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    float scale_factor = 1.0f
);

#endif // CUTLASS_FA_H