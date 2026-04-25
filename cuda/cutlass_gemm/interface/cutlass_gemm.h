#ifndef CUTLASS_GEMM_H
#define CUTLASS_GEMM_H

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

// 使用CUTLASS实现的GEMM操作
template<typename Element>
cudaError_t cutlass_gemm(
    const Element* d_A,
    const Element* d_B,
    Element* d_C,
    int M, int N, int K,
    Element alpha = Element(1), Element beta = Element(0));

#endif // CUTLASS_GEMM_H