from setuptools import find_packages, setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="harl",
    version="1.0.0",
    author="PKU-MARL",
    description="PyTorch implementation of HARL Algorithms",
    url="https://github.com/PKU-MARL/HARL",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "pyyaml>=5.3.1",
        "tensorboard>=2.2.1",
        "tensorboardX",
        "setproctitle",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

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