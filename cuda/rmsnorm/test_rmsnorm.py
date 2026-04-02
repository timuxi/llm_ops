# fixed_test_rmsnorm_cuda.py
# test_rmsnorm_cuda.py
import torch
import torch.nn as nn
import sys
import os

import importlib.util
import glob
so_files = glob.glob('/home/xcy/llm_ops/cuda/rmsnorm/rmsnorm_cuda*.so')
if so_files:
    # 使用第一个匹配的文件
    so_path = so_files[0]
    spec = importlib.util.spec_from_file_location("rmsnorm_cuda", so_path)
    rmsnorm_cuda = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rmsnorm_cuda)
    print(f"✅ Successfully imported rmsnorm_cuda from {os.path.basename(so_path)}")
else:
    print("❌ Could not find rmsnorm_cuda .so file")
    sys.exit(1)
import time
from rmsnorm_module import CustomRMSNorm 


class RMSNorm(nn.Module):
    def __init__(self, weight, eps = 1e-6):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        res = self.weight * hidden_states
        return res.to(input_dtype)

def measure_performance(operation_func, iterations=100):
    """
    测量给定操作的性能
    
    Args:
        operation_func: 要测量的操作函数
        iterations: 执行次数
    
    Returns:
        平均每次操作的时间（毫秒）
    """
    # 计算吞吐量
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for i in range(iterations):  # 执行指定次数以获得更好的统计
        operation_func()
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_per_call_ms = elapsed_time_ms / iterations

    return avg_time_per_call_ms


def test_rmsnorm_performance():
    from torch.profiler import profile, record_function, ProfilerActivity
      # 设置参数
    batch_size, seq_len, hidden_size = 64, 512, 768
    eps = 1e-6
    input_tensor = torch.randn(batch_size * seq_len, hidden_size, dtype=torch.float32, device='cuda')
    weight = torch.randn(hidden_size, dtype=torch.float32, device='cuda')
    grad_output = torch.randn_like(input_tensor)
    input_tensor.requires_grad_(True)
    weight.requires_grad_(True)

    custom_rms_norm = CustomRMSNorm(hidden_size, eps).cuda()
    rms_norm = RMSNorm(hidden_size, eps).cuda()
    
    # forward
    custom_ms = measure_performance(lambda: custom_rms_norm(input_tensor))
    rms_ms = measure_performance(lambda: rms_norm(input_tensor))
    # backward
    torch.cuda.synchronize()
    _1 = custom_rms_norm(input_tensor)
    _2 = rms_norm(input_tensor)
    custom_grad_ms = measure_performance(lambda: _1.backward(grad_output, retain_graph=True))
    rms_grad_ms = measure_performance(lambda: _2.backward(grad_output, retain_graph=True))

    print(f"测试CustomRMSNorm性能: {custom_ms:.4f} ms")
    print(f"测试CustomRMSNorm反向性能: {custom_grad_ms:.4f} ms")
    print(f"测试RMSNorm性能: {rms_ms:.4f} ms")
    print(f"测试RMSNorm反向性能: {rms_grad_ms:.4f} ms")


def test_rmsnorm_cuda():
    # 设置参数
    batch_size, seq_len, hidden_size = 64, 512, 768
    eps = 1e-6
    input_tensor = torch.randn(batch_size * seq_len, hidden_size, dtype=torch.float32, device='cuda')
    weight = torch.randn(hidden_size, dtype=torch.float32, device='cuda')
    grad_output = torch.randn_like(input_tensor)
    input_tensor.requires_grad_(True)
    weight.requires_grad_(True)
    # 测试rmsnorm_forward
    try:
        output_tensor = torch.empty_like(input_tensor)
        rmsnorm_cuda.rmsnorm_forward(input_tensor, weight, output_tensor, eps)
        print("✅ Forward launch successful")
    except Exception as e:
        print(f"❌ Forward launch failed: {e}")
        return False
    # 测试rmsnorm_backward
    try:
        grad_input = torch.empty_like(input_tensor)
        grad_weight = torch.empty_like(weight)
        rmsnorm_cuda.rmsnorm_backward(grad_output, input_tensor, weight, grad_input, grad_weight, eps)
        print("✅ Backward launch successful")
    except Exception as e:
        print(f"❌ Backward launch failed: {e}")
        return False
    
    # golden
    rms_norm = RMSNorm(weight, eps).cuda()
    # forward
    out_golden = rms_norm(input_tensor)
    grad_input = torch.empty_like(input_tensor)
    grad_weight = torch.empty_like(weight)
    grad_output = torch.randn_like(out_golden)
    # backward
    out_golden.backward(grad_output)
    grad_input = input_tensor.grad
    grad_weight = rms_norm.weight.grad

    # compare
    is_close = torch.allclose(output_tensor.cpu(), out_golden.cpu(), rtol=1e-3, atol=1e-3)
    is_close_grad_input = torch.allclose(grad_input.cpu(), input_tensor.grad.cpu(), rtol=1e-3, atol=1e-3)
    is_close_grad_weight = torch.allclose(grad_weight.cpu(), rms_norm.weight.grad.cpu(), rtol=1e-3, atol=1e-3)
    if is_close and is_close_grad_input and is_close_grad_weight:
        print("✅ Forward and backward results match golden implementation")
    else:
        print(is_close, is_close_grad_input, is_close_grad_weight)
        print("❌ Forward and backward results do not match golden implementation")
        return False

    return True

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("✅ CUDA is available")
        success = test_rmsnorm_cuda()
        test_rmsnorm_performance()
        if not success:
            print("\n❌ Tests failed")
        else:
            print("\n🎉 All tests passed")
    else:
        print("❌ CUDA is not available")
    torch.cuda.synchronize()