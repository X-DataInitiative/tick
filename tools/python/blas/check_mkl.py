#!/usr/bin/env python
# -*- coding: utf8 -*-

# python setup.py build_ext --inplace

##
# This file exists to check if the system
#  being used to compile Tick has support
#  for mkl - the "mkl.cpp" file attempts
#  to include "mkl.h" - if it fails we are quite
#  confident that mkl is not installed on the system
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
            Extension("mkl", sources=["tools/python/blas/mkl.cpp"], language="c++",)
          ]

class MKLBuild(build):
    def run(self):
        self.run_command('build_ext')
        build.run(self)

setup(name="checkMKL",
      version='0.6.0.0',
      ext_modules=modules,
      packages=find_packages(),
      cmdclass={'build': MKLBuild
               }
     )
