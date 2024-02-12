#!/usr/bin/env python3

import numpy

from setuptools import setup, find_packages

from distutils import core
from distutils.extension import Extension
from Cython.Build import cythonize


extensions = [
    Extension('cytm.lda_c',   sources=['cytm/lda_c.pyx'],   language="c++", extra_compile_args=["-std=c++11", "-O3", "-mtune=native", "-march=native", "-mfpmath=both"]),
    Extension('cytm.ldab_c',  sources=['cytm/ldab_c.pyx'],  language="c++", extra_compile_args=["-std=c++11", "-O3", "-mtune=native", "-march=native", "-mfpmath=both"]),
    Extension('cytm.atm_c',   sources=['cytm/atm_c.pyx'],   language="c++", extra_compile_args=["-std=c++11", "-O3", "-mtune=native", "-march=native", "-mfpmath=both"]),
    Extension('cytm.pltm_c',  sources=['cytm/pltm_c.pyx'],  language="c++", extra_compile_args=["-std=c++11", "-O3", "-mtune=native", "-march=native", "-mfpmath=both"]),
    Extension('cytm.ctm_c',   sources=['cytm/ctm_c.pyx'],   language="c++", extra_compile_args=["-std=c++11", "-O3", "-mtune=native", "-march=native", "-mfpmath=both"]),
    Extension('cytm.nctm_c',  sources=['cytm/nctm_c.pyx'],  language="c++", extra_compile_args=["-std=c++11", "-O3", "-mtune=native", "-march=native", "-mfpmath=both"]),
    Extension('cytm.sppmi_c', sources=['cytm/sppmi_c.pyx'], language="c++", extra_compile_args=["-std=c++11", "-O3", "-mtune=native", "-march=native", "-mfpmath=both"]),
]

#core.setup(
#  ext_modules=cythonize(extensions),
#  include_dirs=[numpy.get_include(), 'cytm/include']
#)

requires = [
]

setup(
  name='cytm',
  version='0.0.1',
  author='take',
  url='',
  packages=find_packages(),
  ext_modules=cythonize(extensions, annotate=True, language_level=3),
  include_dirs=[numpy.get_include(), 'cytm/include'],
  scripts=[
  ],
  install_requires=requires,
  license='MIT',
  test_suite='test',
  zip_safe=False,
  classifiers=[
    'Operating System :: OS Independent',
    'Environment :: Console',
    'Programming Language :: Python',
    'License :: OSI Approved :: MIT License',
    'Development Status :: Alpha',
    'Intended Audience :: Information Technology',
    'Intended Audience :: Science/Research',
    'Topic :: Utilities',
  ],
  data_files=[
  ]
)
