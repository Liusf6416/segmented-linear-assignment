import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


nvcc_args = ["-O3", "-lineinfo", "--expt-relaxed-constexpr"]
if os.environ.get("CUDAHOSTCXX"):
    nvcc_args.extend(["-ccbin", os.environ["CUDAHOSTCXX"]])

setup(
    ext_modules=[
        CUDAExtension(
            "segmented_linear_assignment._C",
            [
                "src/bindings.cpp",
                "src/grouped_lap_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                # Fast math can change LAP decisions near a tie.
                "nvcc": nvcc_args,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
