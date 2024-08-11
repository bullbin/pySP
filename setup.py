from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

# TODO - Support GCC, this is for MSVC

setup(
    ext_modules=cythonize([Extension("debayer.ahd_homogeneity_cython",
                                     sources=["debayer\\ahd_homogeneity_cython.pyx"],
                                     extra_compile_args=['/openmp:llvm', '/Ox', '/fp:fast'],
                                     include_dirs=[numpy.get_include()]),
                            Extension("dng_warp_corr.dng_warp_rectilinear_coords",
                                     sources=["dng_warp_corr\\dng_warp_rectilinear_coords.pyx"],
                                     extra_compile_args=['/openmp:llvm', '/Ox', '/fp:fast'],
                                     include_dirs=[numpy.get_include()])],
                          compiler_directives={'language_level' : "3"})
)