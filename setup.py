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

extensions = []

GWEButils = Extension("GWEButils_cFunc", 
        sources=[join(path, 'GWEButils_cFunc.pyx')], 
        include_dirs=[np.get_include()], 
        define_macros=defs)
extensions.append(GWEButils)

useGSL = True
try:
    GWEButils_gsl = Extension("GWEButils_cFunc_gsl", 
        sources=[join(path, 'GWEButils_cFunc_gsl.pyx')], 
        include_dirs=[np.get_include(), cython_gsl.get_include()], 
        define_macros=defs, libraries=cython_gsl.get_libraries(), 
        library_dirs=[cython_gsl.get_library_dir()])
    extensions.append(GWEButils_gsl)
except:
    useGSL = False

setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level" : "3"}), 
)
if not useGSL:
    print('Warning: GSL is not installed in the system!')
    print('Numpy is used for scientific computation instead ... The speed is slower than GSL')
