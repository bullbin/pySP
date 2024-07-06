from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

# TODO - Support GCC, this is for MSVC

setup(
    ext_modules=cythonize((Extension("debayer.ahd_homogeneity_cython",
                                     sources=["debayer\\ahd_homogeneity_cython.pyx"],
                                     extra_compile_args=['/openmp', '/Ox'],
                                     include_dirs=[numpy.get_include()])))
)