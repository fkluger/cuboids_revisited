#!/bin/env python
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext = Extension(
            "_sampler",
            [
                "_sampler.pyx",
                "sampling.cpp"
            ],
            language="c++11",
            libraries=["stdc++"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-std=c++11", "-O3"]
        )

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext]
)
