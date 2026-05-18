from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="moe_gate_up_nvfp4_swiglu_cuda",
    ext_modules=[
        CUDAExtension(
            name="moe_gate_up_nvfp4_swiglu_cuda",
            sources=["moe_gate_up_nvfp4_swiglu_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "-gencode=arch=compute_100a,code=sm_100a",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
