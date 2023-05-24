#!/usr/bin/env python3
"""
Build the Cython demonstrations of low-level access to NumPy random

Usage: python setup.py build_ext -i
"""

import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
from os.path import join, dirname
import cython_gsl

path = dirname(__file__)
defs = [('NPY_NO_DEPRECATED_API', 0)]

GWEButils_gsl = Extension("GWEButils_cFunc_gsl", 
        sources=[join(path, 'GWEButils_cFunc_gsl.pyx')], 
        include_dirs=[np.get_include(), cython_gsl.get_include()], 
        define_macros=defs, libraries=cython_gsl.get_libraries(), 
        library_dirs=[cython_gsl.get_library_dir()])

GWEButils = Extension("GWEButils_cFunc", 
        sources=[join(path, 'GWEButils_cFunc.pyx')], 
        include_dirs=[np.get_include()], 
        define_macros=defs)
extensions = [GWEButils, GWEButils_gsl]

setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level" : "3"}), 
)
