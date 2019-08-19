#!/usr/bin/env python

import numpy

from setuptools import setup, find_packages

from distutils import core
from distutils.extension import Extension
from Cython.Build import cythonize


extensions = [
    Extension('model.lda_c',   sources=['model/lda_c.pyx'],   language="c++", extra_compile_args=["-std=c++11", "-O3", "-mtune=native", "-march=native", "-mfpmath=both"]),
    Extension('model.pltm_c',  sources=['model/pltm_c.pyx'],  language="c++", extra_compile_args=["-std=c++11", "-O3", "-mtune=native", "-march=native", "-mfpmath=both"]),
    Extension('model.ctm_c',   sources=['model/ctm_c.pyx'],   language="c++", extra_compile_args=["-std=c++11", "-O3", "-mtune=native", "-march=native", "-mfpmath=both"]),
    Extension('model.nctm_c',  sources=['model/nctm_c.pyx'],  language="c++", extra_compile_args=["-std=c++11", "-O3", "-mtune=native", "-march=native", "-mfpmath=both"]),
]

#core.setup(
#  ext_modules=cythonize(extensions),
#  include_dirs=[numpy.get_include(), 'model/include']
#)

requires = [
]

setup(
  name='topic-model',
  version='0.0.1',
  author='take',
  url='',
  packages=find_packages(),
  ext_modules=cythonize(extensions),
  include_dirs=[numpy.get_include(), 'model/include'],
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
