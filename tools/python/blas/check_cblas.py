#!/usr/bin/env python
# -*- coding: utf8 -*-

# python setup.py build_ext --inplace

##
# This file exists to check if the system
#  being used to compile Tick has support
#  for cblas - the "cblas.cpp" file attempts
#  to include "cbas.h" - if it fails we are quite
#  confident that cblas is not installed on the system
##

"""
setup.py file
"""

import distutils
from distutils import sysconfig
from distutils.version import LooseVersion
from distutils.command.build import build
from distutils.command.clean import clean

from setuptools import find_packages, setup, Command
from setuptools.command.install import install
from setuptools.extension import Extension

modules = [
            Extension("blas", sources=["tools/python/blas/cblas.cpp"], language="c++",)
          ]

class BLASBuild(build):
    def run(self):
        self.run_command('build_ext')
        build.run(self)

setup(name="checkBLAS",
      version='0.5.0.0',
      ext_modules=modules,
      install_requires=['scipy',
                        'numpydoc',
                        'scikit-learn'],
      packages=find_packages(),
      cmdclass={'build': BLASBuild
               }
     )
