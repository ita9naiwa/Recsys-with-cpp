import os
import numpy
from distutils.core import setup, Extension
from Cython.Build import cythonize
numpy_include_dirs = os.path.split(numpy.__file__)[0] + '/core/include'


ext_modules=[Extension(
                "algo.solver",
                sources=["_solver.pyx", "lib/solver.cpp"],
                language='c++',
                include_dirs=["submodule/eigen", "./"] + [numpy_include_dirs],
                extra_compile_args=['-std=c++14']),
            ]


setup(
    name="AlgoPrac",
    version="0.0.1",
    packages=['algo'],
    ext_modules=cythonize(ext_modules))
