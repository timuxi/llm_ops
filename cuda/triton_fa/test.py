import torch
import triton
import triton.testing
import triton.language as tl

# 导入官方 FlashAttention
from flash_attn import flash_attn_func
from kernel.fa_v1 import flash_attn_v1

DEVICE = torch.device('cuda')
torch.set_float32_matmul_precision('high')


# ----------------------------
# 1. 定义 naive attention
# ----------------------------
def naive_attention(q, k, v):
    B, H, N, D = q.shape
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output


# ----------------------------
# 2. 用 torch.compile 包裹
# ----------------------------
compiled_attention = torch.compile(naive_attention, mode="reduce-overhead")


# ----------------------------
# 3. 官方 FlashAttention 封装（适配输入维度）
# ----------------------------
def official_flash_attn(q, k, v):
    """
    适配官方 flash_attn_func 的输入维度：
    - 输入：本代码的 q/k/v 是 [B, H, N, D]
    - 官方要求：query/key/value 是 [B, N, H, D]
    - 输出转换回 [B, H, N, D] 以保持和其他实现一致
    """
    # 维度转置：(B, H, N, D) → (B, N, H, D)
    q_official = q.permute(0, 2, 1, 3)
    k_official = k.permute(0, 2, 1, 3)
    v_official = v.permute(0, 2, 1, 3)

    # 调用官方 FlashAttention（默认无因果掩码，和 naive 实现对齐）
    out_official = flash_attn_func(
        q_official, k_official, v_official,
        dropout_p=0.0,  # 关闭 dropout 以对齐
        causal=False  # 非因果注意力，和 naive 实现对齐
    )

    # 转置回原维度：(B, N, H, D) → (B, H, N, D)
    out_official = out_official.permute(0, 2, 1, 3)
    return out_official


# ----------------------------
# 5. 正确性验证（新增官方 FA 对比）
# ----------------------------
def validate_correctness():
    """验证 Triton FlashAttention / 官方 FlashAttention 与 naive 实现的输出是否一致"""
    torch.manual_seed(0)
    B, H, N, D = 2, 4, 256, 64
    dtype = torch.float16
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)

    with torch.no_grad():
        out_ref = naive_attention(q, k, v)
        out_triton = flash_attn_v1(q, k, v)
        out_official = official_flash_attn(q, k, v)  # 官方 FA 结果

    # 计算 Triton 实现与 naive 的误差
    diff_triton = torch.abs(out_ref - out_triton)
    max_diff_triton = diff_triton.max().item()
    mean_diff_triton = diff_triton.mean().item()

    # 计算官方 FA 与 naive 的误差
    diff_official = torch.abs(out_ref - out_official)
    max_diff_official = diff_official.max().item()
    mean_diff_official = diff_official.mean().item()

    print("=" * 60)
    print(f"[✅ Correctness Check] Naive vs Triton FA V1")
    print(f"  Max diff: {max_diff_triton:.6e}, Mean diff: {mean_diff_triton:.6e}")
    print(f"[✅ Correctness Check] Naive vs Official FlashAttention")
    print(f"  Max diff: {max_diff_official:.6e}, Mean diff: {mean_diff_official:.6e}")
    print("=" * 60)

    # float16 下通常允许 ~1e-2 误差（因在线 softmax 细节差异）
    tolerance = 1e-2
    if max_diff_triton > tolerance:
        print("WARNING: Triton FA has large discrepancy!")
    else:
        print("Triton FA outputs are consistent within tolerance.")

    if max_diff_official > tolerance:
        print("WARNING: Official FA has large discrepancy!")
    else:
        print("Official FA outputs are consistent within tolerance.")


# ----------------------------
# 6. Benchmark 函数（新增官方 FA 对比）
# ----------------------------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[256 * i for i in range(2, 10)],  # N = 512, 768, ..., 2048
        line_arg='provider',
        line_vals=['naive', 'torch_compiled', 'triton_fa_v1', 'flash_attn_official'],
        line_names=['Naive Attention', 'Torch Compiled', 'Triton FA V1', 'Official FlashAttention'],
        styles=[('red', '-'), ('blue', '-'), ('green', '-'), ('purple', '-')],
        ylabel='TFLOP/s',
        plot_name='attention-benchmark-with-official-fa',
        args={
            'B': 4,
            'H': 16,
            'D': 64,
        }
    )
)
def benchmark(B, H, D, N, provider):
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)

    if provider == 'naive':
        fn = lambda: naive_attention(q, k, v)
    elif provider == 'torch_compiled':
        fn = lambda: compiled_attention(q, k, v)
    elif provider == 'triton_fa_v1':
        fn = lambda: flash_attn_v1(q, k, v)
    elif provider == 'flash_attn_official':
        fn = lambda: official_flash_attn(q, k, v)  # 官方 FA 基准
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # 预热（避免首次运行编译开销）
    for _ in range(10):
        fn()

    # 性能测试
    ms = triton.testing.do_bench(fn, warmup=10, rep=5)
    # 计算 FLOPs：注意力的 FLOPs 公式为 4*B*H*N*N*D
    flops = 4.0 * B * H * N * N * D
    tflops = flops / (ms * 1e-3) / 1e12
    return tflops


# ----------------------------
# 7. 运行
# ----------------------------
if __name__ == "__main__":
    # 1. 正确性验证
    validate_correctness()

    # 2. 性能基准测试（可选，注释/取消注释控制）
    # benchmark.run(
    #     show_plots=True,
    #     print_data=True,
    #     save_path='./benchmark'
    # )