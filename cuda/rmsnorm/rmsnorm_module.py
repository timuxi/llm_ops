# fixed_rmsnorm_module.py
# llm_ops/rmsnorm_module.py
import torch
import torch.nn as nn
import importlib.util
import os

# Get the directory where this module file is located
module_dir = os.path.dirname(os.path.abspath(__file__))

# Try to import the compiled CUDA extension
so_path = os.path.join(module_dir, "rmsnorm_cuda.cpython-310-x86_64-linux-gnu.so")
spec = importlib.util.spec_from_file_location("rmsnorm_cuda", so_path)
rmsnorm_cuda = importlib.util.module_from_spec(spec)


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, eps):
        assert input.dtype == weight.dtype, "Input and weight must have the same dtype"
        output = torch.empty_like(input)
        rmsnorm_cuda.rmsnorm_forward(input, weight, output, eps)
        
        # Save necessary tensors for backward pass
        ctx.eps = eps
        ctx.input_dtype = input.dtype 
        ctx.save_for_backward(input, weight)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        eps = ctx.eps
        grad_input = torch.empty_like(input)
        grad_weight = torch.empty_like(weight)
        
        rmsnorm_cuda.rmsnorm_backward(
            grad_output, input, weight, grad_input, grad_weight, eps
        )
        
        return grad_input, grad_weight, None  # No gradient for eps

class CustomRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(CustomRMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        return RMSNormFunction.apply(hidden_states, self.weight.to(hidden_states.dtype), self.eps)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"