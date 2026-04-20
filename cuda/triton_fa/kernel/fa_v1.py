import torch
import triton
import triton.testing
import triton.language as tl


def get_kernel_configs():
    configs = []
    configs.append(triton.Config(
                    {'BLOCK_M': 32, 'BLOCK_N': 32},
                    num_warps=4,
                    num_stages=2
                ))
    # for block_m in [32, 64, 128]:
    #     for block_n in [32, 64, 128]:
    #         # 针对不同硬件特性调整
    #         for num_warps in [4, 8, 16]:
    #             for num_stages in [1, 2, 3]:
    #                 if block_m * block_n <= 16384:  # 避免太大
    #                     configs.append(triton.Config(
    #                         {'BLOCK_M': block_m, 'BLOCK_N': block_n},
    #                         num_warps=num_warps,
    #                         num_stages=num_stages
    #                     ))
    return configs

@triton.autotune(
    configs=get_kernel_configs(),
    key=['M', 'N', ]  # 使用这些变量作为 key，当这些值变化时重新调优
)
@triton.jit
def flash_attn_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        B, H, M, N,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    m_size = tl.cdiv(M, BLOCK_M)
    bn_idx = pid // m_size
    batch_idx = bn_idx // H
    head_idx = bn_idx % H
    m_idx = pid - bn_idx * m_size
    # 加载 Q
    m_range = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_range[:, None] < M

    q_off = (batch_idx * stride_qb + head_idx * stride_qh +
             m_range[:, None] * stride_qm +
             tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qk)
    q = tl.load(q_ptr + q_off, mask=m_mask, other=0.0)
    softmax_scale = 1.0 / (BLOCK_DMODEL ** 0.5)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for n_idx in range(0, N, BLOCK_N):
        n_range = n_idx + tl.arange(0, BLOCK_N)
        n_mask = n_range[None, :] < N
        k_off = (batch_idx * stride_kb + head_idx * stride_kh +
                 n_range[None, :] * stride_kn +
                 tl.arange(0, BLOCK_DMODEL)[:, None] * stride_kk)
        k = tl.load(k_ptr + k_off, mask=n_mask, other=0.0)
        qk = tl.dot(q, k)
        qk *= softmax_scale
        m_ij = tl.max(qk, axis=1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        # 更新全局统计
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        l_new = alpha * l_i + beta * l_ij
        # 更新 Acc
        acc = acc * alpha[:, None]
        # --- 加载 V 块 ---
        v_off = (batch_idx * stride_vb + head_idx * stride_vh +
                 n_range[:, None] * stride_vn +  # n_range 扩展为行 [BLOCK_N, 1]
                 tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vk)  # D_model 扩展为列 [1, BLOCK_D]
        n_range_for_v = n_idx + tl.arange(0, BLOCK_N)
        v_mask = n_range_for_v[:, None] < N
        v = tl.load(v_ptr + v_off, mask=v_mask, other=0.0)
        p = p * beta[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        # 更新状态
        m_i = m_new
        l_i = l_new
    acc = acc / l_i[:, None]
    # 写回结果
    o_off = (batch_idx * stride_ob + head_idx * stride_oh +
             m_range[:, None] * stride_om +
             tl.arange(0, BLOCK_DMODEL)[None, :] * stride_ok)
    tl.store(out_ptr + o_off, mask=m_mask, value=acc)

    
def flash_attn_v1(q, k, v):
    B, H, M, D = q.shape
    N = k.shape[2]
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * B * H,
    )
    o = torch.empty_like(q)

    flash_attn_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, M, N,
        BLOCK_DMODEL=D,
    )
    return o

