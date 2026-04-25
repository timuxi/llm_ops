// /home/xcy/llm_ops/cuda/rmsnorm/rmsnorm_cuda.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include "../kernel/rmsnorm_kernel.h"


// Forward function implementation
template<typename INPUT_TYPE>
void rmsnorm_cuda_forward_impl(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    float eps
) {
    const int n_rows = input.size(0); // First dimension
    const int n_cols = input.size(1); // Second dimension (hidden dimension)

    rmsnorm_forward_float(
        input.data_ptr<INPUT_TYPE>(), 
        weight.data_ptr<INPUT_TYPE>(), 
        output.data_ptr<INPUT_TYPE>(), 
        eps, 
        n_rows, 
        n_cols);
}

template<typename INPUT_TYPE>
void rmsnorm_cuda_backward_impl(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor grad_input,
    torch::Tensor grad_weight,
    float eps
) {
    const int n_rows = input.size(0); // First dimension
    const int n_cols = input.size(1); // Second dimension (hidden dimension)

    rmsnorm_backward_float(
        grad_output.data_ptr<INPUT_TYPE>(), 
        input.data_ptr<INPUT_TYPE>(), 
        weight.data_ptr<INPUT_TYPE>(), 
        grad_input.data_ptr<INPUT_TYPE>(), 
        grad_weight.data_ptr<INPUT_TYPE>(), 
        eps, 
        n_rows, 
        n_cols);
}

void rmsnorm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    float eps
) {
    switch(input.scalar_type()) {
        case at::ScalarType::Float :
            rmsnorm_cuda_forward_impl<float>(input, weight, output, eps);
            break;
        default:
            TORCH_CHECK(false, "RMSNorm only supports float, half, and bfloat16 types, got: ", 
                       input.scalar_type());
    }
}

void rmsnorm_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor grad_input,
    torch::Tensor grad_weight,
    float eps
) {
    switch(input.scalar_type()) {
        case at::ScalarType::Float :
            rmsnorm_cuda_backward_impl<float>(grad_output, input, weight, grad_input, grad_weight, eps);
            break;
        default:
            TORCH_CHECK(false, "RMSNorm only supports float, half, and bfloat16 types, got: ", 
                       input.scalar_type());
    }
}



// PyBind11接口
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_cuda_forward, "RMSNorm forward (CUDA)");
    m.def("rmsnorm_backward", &rmsnorm_cuda_backward, "RMSNorm backward (CUDA)");
}
