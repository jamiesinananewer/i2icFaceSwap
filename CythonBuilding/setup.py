from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

extensions = [
    Extension(
        "SwapCython",                                        # Module Name
        sources = ["lib/swap.pyx", "lib/swap_kernel.cpp"],      # Source files
        language = "c++",                                       # Using C++ 
        include_dirs = [np.get_include()],                      # Include numpy headers
        extra_compile_args=["-std=c++11"],
    )
]

setup(
    name = "SwapCython",
    cmdclass={'build_ext': build_ext},
    ext_modules = cythonize(extensions),
)