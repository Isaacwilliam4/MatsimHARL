from setuptools import find_packages, setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("./harl/envs/flowsim/cython/reward_core.pyx", language_level=3),
    include_dirs=[np.get_include()],
)

setup(
    ext_modules=cythonize(
        "harl/envs/flowsim/cython/cy_bfs.pyx",  # Add the correct path
        language="c++",  # Force C++ compilation
        compiler_directives={'language_level': '3'}
    ),
    include_dirs=[np.get_include()],
    extra_compile_args=["-std=c++11"],  # Ensure C++11 is used
    extra_link_args=["-std=c++11"]
)