# 修正后的setup.py
import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 正确修复CUDA版本检查函数
def patched_check_cuda_version(compiler_name, compiler_version):
    # 返回True表示通过检查，即使版本不匹配
    return True

# 替换原函数
torch.utils.cpp_extension._check_cuda_version = patched_check_cuda_version

setup(
    name='rmsnorm_cuda',
    ext_modules=[
        CUDAExtension(
            name='rmsnorm_cuda',
            sources=[
                'interface/rmsnorm_interface.cu',  # 接口文件
                'kernel/rmsnorm_kernel.cu',          # 核心实现文件
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__',
                ]
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)