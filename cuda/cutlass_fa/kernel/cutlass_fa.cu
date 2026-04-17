#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cutlass/half.h>
#include "cutlass_fa.h"
#include "static_switch.h"
#include "kernel_traits.h"
#include "flash.h"
#include "utils.h"

// data type to test - 使用与接口一致的类型
using FP = cutlass::half_t;
using FPC = cutlass::half_t;  // 与接口声明保持一致
using FPC_O = cutlass::half_t;  // 与接口声明保持一致

const int Bm = 64;
const int Bn = 64;

// TODO: causal模式下, warp!=1情况有bug
// 使用kNThreads
const int Warps = 4;
const bool IS_CAUSAL = false;

const int BS = 2;
const int HEAD = 16;
const int SEQLEN = 128 * 3;
const int DIM = 64;
// const float softmax_scale = 1.f / sqrtf(static_cast<float>(SEQLEN));
const float softmax_scale = 1.f;

// debug only
int TX = 3;
int TY = 0;

// TODO: test trait
// TODO: test trait
using Test_Traits = Flash_fwd_kernel_traits<DIM, Bm, Bn, Warps, FPC>;

namespace flash{
    // Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
    // TODO: 搞清楚经过convert_layout_acc_rowcol后(nrow=(2, MMA_M), ncol=(2, MMA_N))的数学含义
    // 形象的解释是把
    //    T1.V0
    //    T1.V1
    //    T1.V0
    //    T1.V1
    // 变为
    //    T1.V0 T1.V1
    //    T1.V0 T1.V1
    // 这样符合MMA tile的行列直觉
    template<typename Layout>
    inline __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
        // TD [2023-08-13]: Idk why but get<0, 1>(l) doesn't work for Cutlass 3.2, I'm getting
        // "int_tuple.hpp(74): error: conversion to inaccessible base class"
        // return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
        return make_layout(make_layout(get<1>(get<0>(l)), get<1>(l)), make_layout(get<0>(get<0>(l)), get<2>(l)));
    };

    template <int kBlockM, int kBlockN, int kNWarps,typename Engine, typename Layout>
    inline __device__ void mask_within_nblock(Tensor<Engine, Layout> &tensor, const int m_block, const int nbi) {
        // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
        static_assert(Layout::rank == 2, "Only support 2D Tensor");

        // NOTE:
        // 确定一个MMA内的index也是一个难点
        // (nrow=(2, MMA_M), ncol=(2, MMA_N))形如:
        //    T1.V0 T1.V1
        //    T1.V0 T1.V1
        // 根据mma_tile的示意图来确定col和row值

        // NOTE:
        // 计算thread的处理范围, mask掉超出范围的部分
        //
        // NOTE:
        // % 32表示32做组, 因为SM80_16x8x16_F32F16F16F32_TN _1_2_1中最大线程数id是32
        // (lane_id % 4) * 2表示在哪个"颜色"的col(thread)中, *2是为了靠右(即处理的哪个value2)
        // 因此col_idx_offset表示当前thread所处理的单个Atom中4列的哪列

        // lane_id表示一个MMA tile中的"线程组"
        const int lane_id = threadIdx.x % 32;
        const int col_idx_offset = kBlockN * nbi + (lane_id % 4) * 2;

        const int nrow_group = threadIdx.x / 32;
        const int row_idx_offset = kBlockM * m_block + lane_id / 4 + nrow_group * 16 /* 2*8 */;
        // (2, nrow), 2*8 for each
        const int group_stride = kNWarps * 16;

        #pragma unroll
        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
            // SM80_16x8x16_F32F16F16F32_TN中的一组中, 一行4个线程处理8个value
            const int col_idx_base = col_idx_offset + nj * 8;
            #pragma unroll
            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                // j用于计算value 1和value 2对应col
                // col_idx最终表示当前thread所处理的value的列号
                const int col_idx = col_idx_base + j;

                // mask掉scores中(QK后的结果)超出范围的部分
                // 列号和行号对比

                // Without the "make_coord" we get wrong results
                // for nrow(2, MMA_M)
                #pragma unroll
                for (int mi = 0; mi < size<0, 0>(tensor); ++mi) {

                #pragma unroll
                for (int mj = 0; mj < size<0, 1>(tensor); ++mj) {
                    const int row_idx = row_idx_offset + mi * 8 + mj * group_stride;
                    if (col_idx > row_idx) {
                    tensor(make_coord(mi, mj), make_coord(j, nj)) = -INFINITY;
                    }
                }

                }

            }
        }
    }


    template <typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
    inline __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                                Tensor<Engine1, Layout1> &D) {
        CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
        CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
        CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
        CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
        CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

        #pragma unroll
        for (int m = 0; m < size<1>(S); ++m) {
            // TODO: 原版处这里identity_MN是用来跳过大块的block的, predicate用于跳过block内的拷贝
            // TODO: 添加predicate逻辑, 用于跳过无用拷贝
            // if (get<0>(identity_MN(0, m, 0)) < max_MN)
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
            }
        }
    }

    template<typename Tensor0, typename Tensor1,
            typename Tensor2, typename Tensor3, typename Tensor4,
            typename TiledMma, typename TiledCopyA, typename TiledCopyB,
            typename ThrCopyA, typename ThrCopyB>
    inline __device__ void gemm_smem(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                                Tensor4 const& tCsB, TiledMma tiled_mma,
                                TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                                ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
        // NOTE: s -> reg
        cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
        cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
        #pragma unroll
        for (int i = 0; i < size<2>(tCrA); ++i) {
            if (i < size<2>(tCrA) - 1) {
                cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
                cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        }
    }

    // Blocks until all but N previous cp.async.commit_group operations have committed.
    // This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
    // (which is equivalent to commit_group then wait_group 0).
    // Instead we just call cp.async.wait_group 0, which is slightly faster.
    // https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
    template <int N>
    CUTE_HOST_DEVICE
    void cp_async_wait() {
    #if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
        asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
    #endif
    }

    // Apply the exp to all the elements.
    template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
    inline __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
        static_assert(Layout0::rank == 2, "Only support 2D Tensor");
        static_assert(Layout1::rank == 1, "Only support 1D Tensor");
        CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
        #pragma unroll
        for (int mi = 0; mi < size<0>(tensor); ++mi) {
            // If max is -inf, then all elements must have been -inf (possibly due to masking).
            // We don't want (-inf - (-inf)) since that would give NaN.
            // If we don't have float around M_LOG2E the multiplication is done in fp64.
            const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
            #pragma unroll
            for (int ni = 0; ni < size<1>(tensor); ++ni)  {
                // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                // max * log_2(e)) This allows the compiler to use the ffma
                // instruction instead of fadd and fmul separately.
                tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            }
        }
    }

    template<bool Is_first, typename Tensor0, typename Tensor1, typename Tensor2>
    inline __device__ void softmax_rescale_o(Tensor0 &scores, Tensor1 &scores_max, Tensor1 &scores_sum,
                                            Tensor2 &acc_o, float softmax_scale_log2) {
        // NOTE: scores来自acc_s: Q@K.T
        // acc_s用来存储QK和softmax的结果[seqlen, seqlen]
        // acc_o用来存储softmax(QK)结果的分子部分, 用于rescale
        // 流式计算不断用当前分块计算的结果scors来rescale

        if (Is_first) {
            // NOTE: 优化, 第一次softmax不需要rescale, 只需要记录分子, max, sum
            reduce_max</*zero_init=*/true>(scores, scores_max);
            flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
            reduce_sum(scores, scores_sum);
        } else {
            // 记录上一次的max
            Tensor scores_max_prev = make_fragment_like(scores_max);
            cute::copy(scores_max, scores_max_prev);
            // TODO: reduce的实现学习一下
            // NOTE: 计算新max到scores_max
            // reduce_max包含步:
            //  1. 求当前thread内max: 遍历
            //  2. reduce thread间的max: 使用shift技巧reduce
            reduce_max</*zero_init=*/false>(scores, scores_max);
            // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            // 将acc_o转换成符合2D直觉的(nrow, ncol)的形状
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
            #pragma unroll
            for (int mi = 0; mi < size(scores_max); ++mi) {
                // NOTE: 辅助变量: 当前max
                float scores_max_cur = scores_max(mi);
                // NOTE: 计算旧score的rescale值
                // NOTE: 因为QK(影响max)计算时没有考虑softmax_scale, 所以这里要补上
                float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                // NOTE: rescale旧分母部分
                scores_sum(mi) *= scores_scale;
                // NOTE: 旧分子部分rescale
                // acc_o_rowcol.shape = (nrow, ncol)
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
            }
            // NOTE: 计算新分子部分: 对所有scores进行rescale
            flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);

            // NOTE: 累加新分母
            Tensor scores_sum_cur = make_fragment_like(scores_sum);
            // NOTE:利用新分子来累加新分母
            //  1. 线程内累加: 遍历
            //  2. 线程间累加: 使用shift技巧reduce
            reduce_sum(scores, scores_sum_cur);
            // NOTE: 新分母累加到旧分母
            #pragma unroll
            for (int mi = 0; mi < size(scores_sum); ++mi) { scores_sum(mi) += scores_sum_cur(mi); }
        }
    };

    template <typename Fragment>
    inline __device__ auto convert_type_f32_to_f16(Fragment const &acc_fp32) {
    Tensor acc_fp16 = make_tensor<cute::half_t>(shape(acc_fp32));
    {
        Tensor acc_fp32x2 = recast< float2>(acc_fp32);
        Tensor acc_fp16x2 = recast<__half2>(acc_fp16);
        for (int i = 0; i < size(acc_fp32x2); ++i) { acc_fp16x2(i) = __float22half2_rn(acc_fp32x2(i)); }
    }
    return acc_fp16;
    }

    // NOTE: A矩阵已经在寄存器中的gemm封装
    template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
            typename TiledMma, typename TiledCopy, typename ThrCopy>
    inline __device__ void gemm_A_in_regs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                                        TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                        ThrCopy smem_thr_copy_B) {
        // NOTE: 符合M N K描述: A[M, K] @ B[N, K] = C[M, N]
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
        // NOTE: retile 成拷贝需要的大小
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
        cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
        #pragma unroll
        for (int i = 0; i < size<2>(tCrA); ++i) {
            if (i < size<2>(tCrA) - 1) {
                cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        }
    }


    // Convert rowcol_layout from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16, or to ((2, 2, 1), MMA_M, MMA_N) if using m16n8k8.
    template<typename MMA_traits, typename Layout>
    inline __device__ auto convert_layout_rowcol_Aregs(Layout rowcol_layout) {
        using X = Underscore;
        static_assert(decltype(size<0, 0>(rowcol_layout))::value == 2);
        static_assert(decltype(size<1, 0>(rowcol_layout))::value == 2);
        constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
        static_assert(mma_shape_K == 8 || mma_shape_K == 16);
        constexpr int MMA_N_divisor = mma_shape_K == 8 ? 1 : 2;
        auto l = logical_divide(rowcol_layout, Shape<X, Shape<X, Int<MMA_N_divisor>>>{});  // ((2, MMA_M), (2, (2, MMA_N / 2)))
        // TD [2023-08-13]: Same error as above on Cutlass 3.2
        // return make_layout(make_layout(get<1, 0>(l), get<0, 0>(l), get<1, 1, 0>(l)),
        //                    get<0, 1>(l),
        //                    get<1, 1, 1>(l));
        return make_layout(make_layout(get<0>(get<1>(l)), get<0>(get<0>(l)), get<0>(get<1>(get<1>(l)))),
                        get<1>(get<0>(l)),
                        get<1>(get<1>(get<1>(l))));
    };
}
// Shared Storage with Aligned addresses.
template <class ElementType, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV>
struct SharedStorage {
  // TODO: Aligned的话smem的计算是否有问题
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutK>> smem_k;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutV>> smem_v;
};



template <typename Kernel_traits, bool Is_causal=false, typename Params>
__global__ void flash_attention_v2_cutlass_kernel(const Params params) {
    // num_m_block: seqlen group
    const int m_block = blockIdx.x;
    // bs * head
    const int base_id = blockIdx.y;
    // The thread index.
    const int tidx = threadIdx.x;
    const int bs_head_offset = base_id * params.head_stride;

    // TODO: 传入泛型
    // NOTE: 小技巧
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using TiledMMA = typename Kernel_traits::MMA;
    using TiledMMA = typename Kernel_traits::TiledMma;
    using index_t = typename Kernel_traits::index_t;
    using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
    using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
    using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;
    using SmemLayoutVt = typename Kernel_traits::SmemLayoutVtransposed;
    using SmemLayoutVtNoSwizzle = typename Kernel_traits::SmemLayoutVtransposedNoSwizzle;

    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    // Shared memory.
    extern __shared__ char smem_[];
    using SharedStorage = SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>;
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_);

    // global memory tensors for Q, K, V. Layout: (seqlen, dim)
    Tensor Q = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + bs_head_offset),
        make_shape(params.seqlen, params.dim),
        make_stride(params.dim, Int<1>{}));
    Tensor K = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + bs_head_offset),
        make_shape(params.seqlen, params.dim),
        make_stride(params.dim, Int<1>{}));
    Tensor V = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + bs_head_offset),
        make_shape(params.seqlen, params.dim),
        make_stride(params.dim, Int<1>{}));
    // global memory tensors for Q, K, V. Layout: (kBlockM, kHeadDim, num_tile_n)
    Tensor gQ = local_tile(Q, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));
    Tensor gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
    Tensor gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
    // shared memory tensors for Q, K, V. Layout: (8, kBlockKSmem, num_tile_n)
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    // Layout: (1, kBlockKSmem)
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});
    // Tensor for V Transpose; used in GEMM-II.
    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});
    Tensor sVtNoSwizzle = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVtNoSwizzle{});

    // 获取MMA抽象
    TiledMMA tiled_mma;
    // NOTE: 准备拷贝Q, K, V到smem的copy对象
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // 流水线加载初始Q, K
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0));
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    flash::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
    flash::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
    cute::cp_async_fence();

    Tensor rAccOut = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    // NOTE: 1. mask between N BLOCKs if is causal mode
    int seqlen_start = m_block * kBlockM;
    int seqlen_end = (m_block + 1) * kBlockM;
    int n_block_max = Is_causal ? cute::ceil_div(seqlen_end, kBlockN) : cute::ceil_div(params.seqlen, kBlockN);
    Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<2 * size<1>(rAccOut)>>{});
    Tensor scores_sum = make_fragment_like(scores_max);
    auto thr_mma = tiled_mma.get_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ); // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK); // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle); // (MMA, MMA_K,MMA_N)
    clear(rAccOut);
    for (int nbi = 0; nbi < n_block_max; nbi++) {
        auto rAccScore = partition_fragment_C(tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));
        clear(rAccScore);
        flash::cp_async_wait<0>();
        __syncthreads();

        // load V
        gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi, _));
        tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
        flash::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
        cute::cp_async_fence();

        // compute Q@K^T
        flash::gemm_smem(rAccScore, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        Tensor scores = make_tensor(rAccScore.data(), flash::convert_layout_acc_rowcol(rAccScore.layout()));
        if (Is_causal ==  true && nbi * kBlockN >= seqlen_start) {
            flash::mask_within_nblock<kBlockM, kBlockN, kNWarps>(scores, m_block, nbi);
        }
        flash::cp_async_wait<0>();
        __syncthreads();

        // copy next K
        if (nbi != n_block_max - 1) {
            gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi + 1, _));
            tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
            flash::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
            cute::cp_async_fence();
        }

        // 计算softmax
        nbi == 0 ? flash::softmax_rescale_o</*Is_first=*/true>(scores, scores_max, scores_sum, rAccOut, params.softmax_scale) :
                flash::softmax_rescale_o</*Is_first=*/false>(scores, scores_max, scores_sum, rAccOut, params.softmax_scale);

        Tensor rP = flash::convert_type_f32_to_f16(rAccScore);
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_rowcol_Aregs<TiledMMA>(scores.layout()));

        // compute P@V
        flash::gemm_A_in_regs(rAccOut, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    Tensor acc_o_rowcol = make_tensor(rAccOut.data(), flash::convert_layout_acc_rowcol(rAccOut.layout()));
    // for row
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
        float sum = scores_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        float scale = inv_sum;
        // for col
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { 
            acc_o_rowcol(mi, ni) *= scale;  
        }
    }
        
    Tensor rO = flash::convert_type_f32_to_f16(rAccOut);
    // 复用sQ的smem做sO的拷出
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    // Tensor O = make_tensor(
    //     make_gmem_ptr(reinterpret_cast<Element *>(params.out_ptr) + bs_head_offset),
    //     make_shape(params.seqlen, params.dim),
    //     make_stride(params.dim, Int<1>{}));
    // Tensor gO = local_tile(O, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));

    // // 创建到smem -> gmem的拷贝
    // typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    // auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    // Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    // Tensor tOgO = gmem_thr_copy_O.partition_D(gO(_, _, 0));
    // __syncthreads();

    // Tensor tOrO = make_tensor<Element>(shape(tOgO));
    // cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
    // flash::copy(gmem_tiled_copy_O, tOrO, tOgO);
}

// FlashAttention前向传播的实现
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
    float scale_factor) {
    
    using Kernel_traits = Test_Traits;
    using Element = typename Kernel_traits::Element;
    using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
    using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
    using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;

    // Q smem size + KV smem size
    constexpr int kSmemSize = Kernel_traits::kSmemSize;

    int bs_stride = num_heads * seq_len * head_dim;
    int head_stride = seq_len * head_dim;
    int seqlen_stride = head_dim;
    int dim_stride = 1;
    // int smem_size = kSmemSize;
    constexpr size_t smem_size = size_t(sizeof(SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>));

    Flash_fwd_params params;
    set_params_fprop(params, batch_size, num_heads, seq_len, head_dim, bs_stride, head_stride,
                    seqlen_stride, dim_stride, (void*)query, (void*)key, (void*)value, (void*)output, scale_factor);

    const int num_m_block =
        (params.seqlen + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    // 设置网格和块维度
    dim3 grid(num_m_block, params.bs * params.head, 1);
    dim3 block(size(Kernel_traits::kNThreads));

    auto kernel = &flash_attention_v2_cutlass_kernel<Kernel_traits, IS_CAUSAL, Flash_fwd_params>;
    // NOTE: smem过大时需要设置
    if (smem_size >= 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    kernel<<<grid, block, smem_size>>>(params);
    CUDA_CHECK(cudaGetLastError());

    cudaError_t err = cudaDeviceSynchronize();
    // 检查CUDA错误
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in FlashAttention forward: " << cudaGetErrorString(err) << std::endl;
    }
    return err;
}

// FlashAttention反向传播的简化实现
cudaError_t cutlass_flash_attention_backward(
    const cutlass::half_t* grad_output,
    const cutlass::half_t* query,
    const cutlass::half_t* key,
    const cutlass::half_t* value,
    const cutlass::half_t* output,
    const float* softmax_lse,
    cutlass::half_t* grad_query,
    cutlass::half_t* grad_key,
    cutlass::half_t* grad_value,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    float scale_factor) {
    
    // 简化的反向传播实现
    int total_elements = batch_size * seq_len * num_heads * head_dim;
    
    // 初始化梯度为0
    cudaMemset(grad_query, 0, total_elements * sizeof(cutlass::half_t));
    cudaMemset(grad_key, 0, total_elements * sizeof(cutlass::half_t));
    cudaMemset(grad_value, 0, total_elements * sizeof(cutlass::half_t));
    
    // 实际的反向传播会非常复杂，这里只作示意
    
    return cudaSuccess;
}