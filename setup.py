from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name="ampere_paged_attention", version="0.1", packages=["ampere_paged_attention"],
      ext_modules=[CUDAExtension("ampere_paged_attention._C", ["csrc/paged_decode.cu"],
                                 extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math", "-lineinfo"]})],
      cmdclass={"build_ext": BuildExtension})
