#ifndef RMSNORM_KERNEL_H
#define RMSNORM_KERNEL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

// C接口函数声明
void rmsnorm_forward_float(const float* input, const float* weight, float* output, 
                        float eps, int n_rows, int n_cols);

// bwd kernel
void rmsnorm_backward_float(const float* grad_output, const float* input, const float* weight,
                            float* grad_input, float* grad_weight, float eps, int n_rows, int n_cols);
#ifdef __cplusplus
}
#endif

#endif // RMSNORM_KERNEL_H