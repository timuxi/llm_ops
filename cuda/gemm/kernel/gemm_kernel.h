#ifndef GEMM_H
#define GEMM_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

// C接口函数声明
void gemm_kernel_float(const float* A, const float* B, float* C, int M, int N, int K);

#ifdef __cplusplus
}
#endif

#endif // GEMM_H
