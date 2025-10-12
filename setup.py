import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from config import kernels

this_dir = os.getcwd()
tk_include_dir = os.path.join(this_dir, 'ThunderKittens', 'include')
tk_prototype_dir = os.path.join(this_dir, 'ThunderKittens', 'prototype')

cuda_flags = [
    '-DNDEBUG',
    '-Xcompiler=-Wno-psabi',
    '-Xcompiler=-fno-strict-aliasing',
    '--expt-extended-lambda',
    '--expt-relaxed-constexpr',
    '-forward-unknown-to-host-compiler',
    '--use_fast_math',
    '-O3',
    '-Xcompiler=-fopenmp',
    '-std=c++20',
    '-DKITTENS_HOPPER',
    '-DTORCH_COMPILE',
    '-arch=sm_90a',
    f'-I{tk_include_dir}',
    f'-I{tk_prototype_dir}',
]
cpp_flags = [
    '-std=c++20',
    '-O3'
]

source_files = ['bindings.cpp']
for kernel_name in kernels:
    kernel_file = f'kernels/{kernel_name}/h100.cu'
    if not os.path.exists(kernel_file):
        raise FileNotFoundError(f"Kernel file not found: {kernel_file}")
    source_files.append(kernel_file)
    cpp_flags.append(f'-DCOMPILE_{kernel_name.upper()}')

setup(
    name='kp',
    ext_modules=[
        CUDAExtension(
            'kp',
            sources=source_files,
            extra_compile_args={
                'cxx': cpp_flags,
                'nvcc': cuda_flags
            },
            libraries=['cuda']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)