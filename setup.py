#!/usr/bin/env python
# -*- coding: utf8 -*-

# python setup.py build_ext --inplace

"""
setup.py file
"""
import multiprocessing
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import time
import unittest
import warnings

from abc import ABC

import distutils
from distutils import sysconfig
from distutils.version import LooseVersion
from distutils.command.build import build
from distutils.command.clean import clean

from setuptools import find_packages, setup, Command
from setuptools.command.install import install
from setuptools.extension import Extension

# Available debug flags
#
#   DEBUG_C_ARRAY       : count #allocations of C-arrays
#   DEBUG_ARRAY         : Track creation/destruction of Array objects
#   DEBUG_SHAREDARRAY   : Track creation/destruction of SharedArray objects
#   DEBUG_VARRAY        : Track VArray
#   DEBUG_COSTLY_THROW  : Enables some costly tests to throw error
#                         (such as Array[i] if i not in range)
#   DEBUG_VERBOSE       : Error messages from CPP extensions will include
#                         backtrace and error loc

# debug_flags = ['DEBUG_C_ARRAY', 'DEBUG_ARRAY', 'DEBUG_COSTLY_THROW',
#                'DEBUG_SHAREDARRAY', 'DEBUG_VARRAY', 'DEBUG_VERBOSE']


debug_flags = ['DEBUG_COSTLY_THROW']

# If true, add compilation flags to use fast (but maybe inaccurate) math
# See https://gcc.gnu.org/wiki/FloatingPointMath
use_fast_math = True

version_info = sys.version_info

python_min_ver = (3, 4, 0)
python_ver = (version_info.major, version_info.minor, version_info.micro)

if python_ver < python_min_ver:
    txt = 'Python version {0}.{1}.{2} ' \
          'lower than the required version >= {3}.{4}.{5}.'

    warnings.warn(txt.format(*(python_ver + python_min_ver)))

# The next block ensures that we build a link-time linkable dynamic library for
# OSX builds instead of a bundle.
#
# Snippet from http://stackoverflow.com/a/32765319/2299947
if sys.platform == 'darwin':
    vars = sysconfig.get_config_vars()
    vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-dynamiclib')

# If we're installing via a wheel or not
is_building_tick = any(arg in ("build",
                               "build_ext",
                               "bdist",
                               "bdist_wheel",
                               "develop",) for arg in sys.argv)

# Obtain the numpy include directory.
# This logic works across numpy versions.
numpy_available = False
numpy_include = ""
blas_info = {}
try:
    import numpy as np
    from numpy.distutils.system_info import get_info

    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()

    # Determine if we have an available BLAS implementation
    if platform.system() == 'Windows':
        try:
            with open(os.devnull, 'w') as devnull:
                exitCode = subprocess.check_output(
                    "python tools/python/blas/check_cblas.py build_ext",
                    stderr=devnull,
                    shell=True)
                blas_info = get_info("blas_opt", 0)
        except subprocess.CalledProcessError as subError:
            print("Error executing check_cblas.py - cblas not found")
    elif 'bdist_wheel' not in sys.argv:
        blas_info = get_info("blas_opt", 0)

    numpy_available = True
except ImportError as e:
    if is_building_tick:
        print(e)
        warnings.warn("numpy is not installed:\n"
                      " - Include directory for numpy integration may not be "
                      "correct\n "
                      " - BLAS will not be used for this build\n")

# By default, we assume that scipy uses 32 bit integers for indices in sparse
# arrays
sparse_indices_flag = "-DTICK_SPARSE_INDICES_INT32"
try:
    import numpy as np
    from scipy.sparse import sputils

    sparsearray_type = sputils.get_index_dtype()

    if sparsearray_type == np.int64:
        sparse_indices_flag = "-DTICK_SPARSE_INDICES_INT64"
except ImportError as e:
    if is_building_tick and numpy_available:
        print(e)
        warnings.warn("scipy is not installed, unable to determine "
                      "sparse array integer type (assuming 32 bits)\n")

if os.name == 'posix':
    if platform.system() == 'Darwin':
        os_version = platform.mac_ver()[0]
        # keep only major + minor
        os_version = '.'.join(os_version.split('.')[:2])

        if LooseVersion(os_version) < LooseVersion('10.9'):
            raise ValueError(
                'You need to have at least mac os 10.9 to build this package')

        # We set this variable manually because anaconda set it to a deprecated
        # one
        os.environ['MACOSX_DEPLOYMENT_TARGET'] = os_version

# Directory containing built .so files before they are moved either
# in source (with build flag --inplace) or to site-packages (by install)
#
# E.g. build/lib.macosx-10.11-x86_64-3.4
build_dir = "build/lib.{}-{}".format(distutils.util.get_platform(),
                                     sys.version[0:3])


class SwigExtension(Extension):
    """This only adds information about extension construction, useful for
    library sharing
    """

    def __init__(self, *args, module_ref=None, ext_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.module_ref = module_ref
        self.ext_name = ext_name


class SwigPath:
    """Small class to handle module creation and check project structure
    """

    def __init__(self, module_path, extension_name):
        module_path = os.path.normpath(module_path)


        # Module C++ source directory (e.g. lib/cpp/tick/base)
        self.src = os.path.join(module_path, 'src')

        # Module SWIG interface files directory (e.g. tick/array/swig)
        self.swig = "lib/swig/" + module_path[5:]

        # Module build directory. Will contain generated .py files, and .so
        # files if built with flag --inplace.
        #
        # E.g. tick/array/build
        self.build = os.path.join(module_path, 'build')

        self.extension_name = extension_name
        self.private_extension_name = '_' + extension_name

        # Transform folder path to module path
        self.extension_path = self.build \
                                  .replace('.', '') \
                                  .replace('/', '.') \
                              + '.' + self.private_extension_name

        self._check_configuration()

        # Filename of the produced .so file (e.g. _array.so)
        self.lib_filename = '{}{}'.format(self.private_extension_name,
                                          sysconfig.get_config_var('SO'))

        print("self.extension_path ", self.extension_path)
        print("self.build ", self.build)
        print("self.lib_filename ", self.lib_filename)

    def _check_configuration(self):
        exception_base_msg = (
            "Swig directory structure must follow this structure :\n"
            "├── module_path\n"
            "    ├── src\n"
            "    │   ├── file.h\n"
            "    │   └── file.cpp\n"
            "    ├── swig\n"
            "    │   └── file.i\n"
            "    └── build\n"
            "        └── generated_files.*\n"
        )
        exception_missing_directory_msg = "%s folder was missing"
        exception_file_instead_directory_msg = "%s should be a directory, " \
                                               "not a file"

        # # Check that src and swig folders do exists
        # for directory in [self.src, self.swig]:
        #     if not os.path.exists(directory):
        #         raise FileNotFoundError(exception_base_msg + (
        #             exception_missing_directory_msg % directory))
        #     elif not os.path.isdir(directory):
        #         raise NotADirectoryError(exception_base_msg + (
        #             exception_file_instead_directory_msg % directory))

        # # Check that build is a directory (not a file) or create it
        # if not os.path.exists(self.build):
        #     os.mkdir(self.build)
        # elif not os.path.isdir(self.build):
        #     raise NotADirectoryError(exception_base_msg + (
        #         exception_file_instead_directory_msg % self.build))


def create_extension(extension_name, module_dir,
                     cpp_files, h_files, swig_files, folders=[],
                     include_modules=None, extra_compile_args=None,
                     swig_opts=None):
    swig_path = SwigPath(module_dir, extension_name)
    extension_path = swig_path.extension_path

    # Add directory to filenames
    def add_dir_name(dir_name, filenames):
        return list(os.path.join(dir_name, filename) for filename in filenames)

    swig_files = add_dir_name("lib/swig/" + module_dir[7:], swig_files)

    for folder in folders:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(file)
                if ext == '.cpp':
                    cpp_files += [os.path.join(folder, file)]
                elif ext == ".txt":
                    pass
                elif ext == ".inl":
                    pass
                else:
                    warnings.warn('Included file %s in folder %s has an '
                                  'unknown extension "%s"' % (file,
                                                              folder,
                                                              ext))

    min_swig_opts = ['-py3',
                     '-c++',
                     '-modern',
                     '-new_repr',
                     '-Ilib/swig/base',
                     '-Ilib/include',
                     '-outdir', swig_path.build,
                     ]

    if swig_opts is None:
        swig_opts = min_swig_opts
    else:
        swig_opts.extend(min_swig_opts)

    # Here we set the minimum compile flags.
    min_extra_compile_args = ["-D_FILE_OFFSET_BITS=64",
                              "-DPYTHON_LINK",
                              "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
                              '-Ilib/include',
                              sparse_indices_flag,
                              '-std=c++11',
                              '-O3',
                              ]

    if use_fast_math:
        min_extra_compile_args.append('-ffast-math')

    if extra_compile_args is None:
        extra_compile_args = min_extra_compile_args
    else:
        extra_compile_args.extend(min_extra_compile_args)

    extra_compile_args.append("-Wall")

    if platform.system() == 'Windows':
        extra_compile_args.append("-DBUILDING_DLL")
    else:
        ## Added -Wall to get all warnings and -Werror to treat them as errors
        ## GITHUB ISSUE: 82 - temporary fix to avoid compile errors from
        ##  warnings given in CEREAL files
        # extra_compile_args.append("-Werror")
        ## This warning is turned off because SWIG generates files that triggers the
        ## warning
        extra_compile_args.append("-Wno-uninitialized")

    # Include directory of module
    mod = SwigPath(module_dir, extension_name)
    for opts in [swig_opts, extra_compile_args]:
        opts.extend(["-Isrc/swig/"+extension_name])

    libraries = []
    library_dirs = []
    runtime_library_dirs = []
    extra_link_args = []
    define_macros = []
    extra_include_dirs = ["include"]

    # Deal with (optional) BLAS
    extra_compile_args.extend(blas_info.get("extra_compile_args", []))
    extra_link_args.extend(blas_info.get("extra_link_args", []))
    libraries.extend(blas_info.get("libraries", []))
    library_dirs.extend(blas_info.get("library_dirs", []))
    define_macros.extend(blas_info.get("define_macros", []))

    if 'define_macros' in blas_info and \
            any(key == 'HAVE_CBLAS' for key, _ in blas_info['define_macros']):
        define_macros.append(('TICK_CBLAS_AVAILABLE', None))

    if include_modules is None:
        include_modules = []

    # Include all what need for module link
    for mod in include_modules:
        if mod.__class__ != SwigPath:
            raise ValueError("Unhandled class for included module")

        for opts in [swig_opts, extra_compile_args]:
            opts.extend(["-I" + mod.swig])

        # Because setuptools produces shared object files with non-standard
        # On windows we need to use ".lib" rather than ".pyd"
        # when linking libs to other libs
        if platform.system() == 'Windows':
            lib = os.path.join(build_dir, mod.build, "_"+mod.extension_name)
            lib += os.path.splitext(sysconfig.get_config_var("SO"))[0]
            libraries.append(lib)
        # names (i.e. not lib<name>.so) we specify the full library path
        else:
            extra_link_args.append(os.path.abspath(
                os.path.join(build_dir, mod.build, mod.lib_filename)))

        # Make sure that the runtime linker can find shared object
        # dependencies by using the relative path to the dependency library.
        rel_path = os.path.relpath(mod.build, swig_path.build)
        if platform.system() == 'Linux':
            # $ORIGIN refers to the location of the current shared object file
            # at runtime
            runtime_library_dirs.append("\$ORIGIN/%s" % rel_path)
        elif platform.system() == 'Windows':
            pass
        else:  # Assuming non-Windows builds for now
            # For OSX builds we use @loader_path instead
            extra_link_args.append(
                "-Wl,-rpath,%s" % '@loader_path/%s' % rel_path
            )

    # Setting the SONAME/install_name for the built libraries. It ensures that
    # the runtime linker will have a chance to find the libraries even after
    # they're moved (during install, for example)
    filename = swig_path.lib_filename
    if platform.system() == 'Linux':
        extra_link_args.append('-Wl,-soname,%s' % filename)
    elif platform.system() == 'Windows':
        pass
    else:
        # For OSX the install_name needs to be prefixed with @rpath
        extra_link_args.append('-Wl,-install_name,@rpath/%s' % filename)

    for df in debug_flags:
        full_flag = "-D" + df

        extra_compile_args.append(full_flag)

        if df == 'DEBUG_COSTLY_THROW':
            swig_opts.append(full_flag)

    # Adding Cereal serialization library
    extra_include_dirs.append("lib/third_party/cereal/include")

    # Adding numpy include directory
    if numpy_include:
        extra_include_dirs.append(numpy_include)

    # This is to override the use of IMPLIB in distutils
    # which puts the lib/exp files in the wrong directory
    # see: https://github.com/python/cpython/blob/08bb8a41cc976343795bd0e241cd7388e9f44ad5/Lib/distutils/_msvccompiler.py#L467
    if platform.system() == 'Windows':
        implib = "/IMPLIB:" + os.path.abspath(
            os.path.join(build_dir, swig_path.build, "_"+extension_name))
        implib += os.path.splitext(sysconfig.get_config_var("SO"))[0]
        extra_link_args.append(implib + ".lib")

    print("extension_path ", extension_path)

    core_module = SwigExtension(extension_path, module_ref=swig_path,
                                sources=swig_files + cpp_files,
                                extra_compile_args=extra_compile_args,
                                extra_link_args=extra_link_args,
                                define_macros=define_macros,
                                swig_opts=swig_opts,
                                libraries=libraries,
                                include_dirs=extra_include_dirs,
                                library_dirs=library_dirs,
                                runtime_library_dirs=runtime_library_dirs,
                                depends=h_files,
                                language="c++",
                                ext_name=extension_name)

    return core_module


##############################
# Create extensions
##############################

array_extension_info = {
    "cpp_files": [],
    "h_files": ["lib/cpp/array"],
    "swig_files": ["array_module.i"],
    "module_dir": "./tick/array/",
    "extension_name": "array"
}

array_extension = create_extension(**array_extension_info)

base_extension_info = {
    "cpp_files": [],
    "h_files": [],
    "folders": [
        "lib/cpp/base",
        "lib/cpp/base/math"
    ],
    "swig_files": ["base_module.i"],
    "module_dir": "./tick/base",
    "extension_name": "base",
    "include_modules": [array_extension.module_ref]
}

base_extension = create_extension(**base_extension_info)

base_array_modules = [array_extension.module_ref, base_extension.module_ref]

array_test_extension_info = {
    "cpp_files": [],
    "h_files": [],
    "folders": ["lib/cpp/array_test"],
    "swig_files": ["array_test_module.i"],
    "module_dir": "./tick/array_test/",
    "extension_name": "array_test",
    "include_modules": base_array_modules,
}

test_extension = create_extension(**array_test_extension_info)

random_extension_info = {
    "cpp_files": [],
    "h_files": [],
    "folders": ["lib/cpp/random"],
    "swig_files": ["crandom_module.i"],
    "module_dir": "./tick/random/",
    "extension_name": "crandom",
    "include_modules": base_array_modules
}

random_extension = create_extension(**random_extension_info)

base_model_core_info = {
    "cpp_files": [],
    "h_files": [],
    "folders": [
        "lib/cpp/base_model"
    ],
    "swig_files": ["base_model_module.i"],
    "module_dir": "./tick/base_model/",
    "extension_name": "base_model",
    "include_modules": base_array_modules
}
base_model_core = create_extension(**base_model_core_info)

linear_model_core_info = {
    "cpp_files": [],
    "h_files": [],
    "folders": [
        "lib/cpp/linear_model"
    ],
    "swig_files": ["linear_model_module.i"],
    "module_dir": "./tick/linear_model/",
    "extension_name": "linear_model",
    "include_modules": base_array_modules + 
    [
      base_model_core.module_ref,
    ]
}
linear_model_core = create_extension(**linear_model_core_info)

hawkes_extension_info = {
    "cpp_files": [],
    "h_files": [],
    "folders": [
        "lib/cpp/hawkes/model",
        "lib/cpp/hawkes/model/base",
        "lib/cpp/hawkes/model/variants",
        "lib/cpp/hawkes/inference",
        "lib/cpp/hawkes/simulation", 
        "lib/cpp/hawkes/simulation/hawkes_baselines", 
        "lib/cpp/hawkes/simulation/hawkes_kernels"
    ],
    "swig_files": [
      "hawkes_module.i"
    ],
    "swig_opts": ["-Ilib/swig/hawkes"],
    "module_dir": "./tick/hawkes/",
    "extension_name": "hawkes",
    "include_modules": base_array_modules + 
    [
      base_model_core.module_ref,
      random_extension.module_ref
    ]
}
hawkes_extension = create_extension(**hawkes_extension_info)

prox_core_info = {
    "cpp_files": [],
    "h_files": [],
    "folders": [
        "lib/cpp/prox"
    ],
    "swig_files": ["prox_module.i"],
    "module_dir": "./tick/prox/",
    "extension_name": "prox",
    "include_modules": base_array_modules
}
prox_core = create_extension(**prox_core_info)

solver_core_info = {
    "cpp_files": [],
    "h_files": [],
    "folders": [
        "lib/cpp/solver"
    ],
    "swig_files": ["solver_module.i"],
    "module_dir": "./tick/solver/",
    "extension_name": "solver",
    "include_modules": base_array_modules + [random_extension.module_ref,
                                             base_model_core.module_ref,
                                             linear_model_core.module_ref,
                                             prox_core.module_ref]
}
solver_core = create_extension(**solver_core_info)

preprocessing_core_info = {
    "cpp_files": [],
    "h_files": [],
    "folders": [
        "lib/cpp/preprocessing"
    ],
    "swig_files": ["preprocessing_module.i"],
    "module_dir": "./tick/preprocessing/",
    "extension_name": "preprocessing",
    "include_modules": base_array_modules
}

preprocessing_core = create_extension(**preprocessing_core_info)

robust_extension_info = {
    "cpp_files": [],
    "h_files": [],
    "folders": [
        "lib/cpp/robust"
    ],
    "swig_files": ["robust_module.i"],
    "module_dir": "./tick/robust/",
    "extension_name": "robust",
    "include_modules": base_array_modules + [
      base_model_core.module_ref,linear_model_core.module_ref]
}
robust_extension = create_extension(**robust_extension_info)

survival_extension_info = {
    "cpp_files": [],
    "h_files": [],
    "folders": [
        "lib/cpp/survival"
    ],
    "swig_files": ["survival_module.i"],
    "module_dir": "./tick/survival/",
    "extension_name": "survival",
    "include_modules": base_array_modules + [base_model_core.module_ref]
}
survival_extension = create_extension(**survival_extension_info)

tick_modules = [
  array_extension, base_extension, test_extension, 
  random_extension, base_model_core, linear_model_core,
  hawkes_extension, prox_core, solver_core, preprocessing_core, 
  robust_extension,  survival_extension 
]

# Abstract class for tick-specific commands that need access to common build
# directories
class TickCommand(Command, ABC):
    tick_dir = os.path.abspath(os.path.join(os.curdir, 'tick'))
    cpp_build_dir = os.path.abspath(os.path.join(build_dir, 'cpptest'))

    user_options = []

    def initialize_options(self):
        """Set default values for options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass


class TickBuild(build):
    swig_min_ver = (3, 0, 7)

    @staticmethod
    def extract_swig_version(swig_ver_str):
        m = re.search('SWIG Version (\d+).(\d+).(\d+)', swig_ver_str)

        if not m:
            txt = 'Could not extract SWIG version from string: {0}'

            warnings.warn(txt.format(swig_ver_str))

            return 0, 0, 0

        return tuple(int(x) for x in m.groups()[0:3])

    def run(self):
        swig_ver = self.extract_swig_version(
            str(subprocess.check_output(['swig', '-version'])))

        if swig_ver < self.swig_min_ver:
            txt = 'SWIG version {0}.{1}.{2} ' \
                  'lower than the required version >= {3}.{4}.{5}. ' \
                  'This will likely cause build errors!'

            warnings.warn(txt.format(*(swig_ver + self.swig_min_ver)))

        self.run_command('build_ext')
        build.run(self)


class TickInstall(install):
    def run(self):
        self.run_command('build_ext')
        install.run(self)


class BuildRunCPPTests(TickCommand):
    description = 'build and run tick C++ tests'

    def run(self):
        self.run_command('makecpptest')
        self.run_command('runcpptest')


class RunCPPTests(TickCommand):
    description = 'run tick C++ tests'

    def run(self):
        make_cmd = ['make', 'check']
        subprocess.check_call(make_cmd, cwd=self.cpp_build_dir)


class BuildCPPTests(TickCommand):
    build_jobs = 1
    description = 'build tick C++ tests'
    user_options = [
        ('build-jobs=', 'j',
         'number of parallel build jobs (default is number of available CPU '
         'cores reported by Python)'),
    ]

    def initialize_options(self):
        """Set default values for options."""
        self.build_jobs = multiprocessing.cpu_count()

    def run(self):
        relpath = os.path.relpath(self.tick_dir, self.cpp_build_dir)
        cmake_exe = os.environ.get('TICK_CMAKE', 'cmake')

        inc_dir = sysconfig.get_python_inc()
        cmake_cmd = [cmake_exe,
                     '-DPYTHON_INCLUDE_DIR={}'.format(inc_dir),
                     '-DTICK_REBUILD_LIBS=OFF',
                     relpath + '/../lib']

        # Feed the path to the built C++ extensions so CMake does not have to
        # build them again
        for mod in tick_modules:
            full_path = os.path.abspath(
                os.path.join(mod.module_ref.build, mod.module_ref.lib_filename))

            cmake_cmd.append(
                '-DTICK_LIB_{}={}'.format(mod.ext_name.upper(), full_path))

        os.makedirs(os.path.join(self.cpp_build_dir, 'cpptest'), exist_ok=True)
        subprocess.check_call(cmake_cmd, cwd=self.cpp_build_dir)

        make_cmd = ['make', 'VERBOSE=1', 'all', '-j{}'.format(self.build_jobs)]
        subprocess.check_call(make_cmd, cwd=self.cpp_build_dir)


class RunCPPLint(TickCommand):
    description = 'run cpplint on tick C++ source files'

    CPPLINT_DIRS = [
        'src/cpp',
    ]

    def run(self):
        try:
            import cpplint as cl

            cl_state = cl._cpplint_state
            error_count = 0

            for dir in self.CPPLINT_DIRS:
                print("Processing {}".format(dir))

                cl_state.ResetErrorCounts()
                filenames = list(pathlib.Path(dir).glob('**/*.h')) + \
                            list(pathlib.Path(dir).glob('**/*.cpp'))

                for filename in filenames:
                    cl.ProcessFile(str(filename), cl_state.verbose_level)
                cl_state.PrintErrorCounts()

                error_count += cl_state.error_count
                print('')

            if error_count > 0:
                raise RuntimeError("Codestyle check by cpplint failed")

        except ImportError:
            warnings.warn("Stylecheck by cpplint failed because cpplint "
                          "is not installed as a Python module")


class RunPyLint(TickCommand):
    # We need to define if and how we run pylint

    description = 'run tick PyLint codestyle check'
    start_dir = '.'

    @staticmethod
    def run():
        raise NotImplementedError('Running pylint from setup.py'
                                  'not supported yet')


class RunPyTests(TickCommand):
    description = 'run tick Python tests'
    start_dir = '.'

    user_options = [
        ('start-dir=', 's',
         'directory to start looking for Python tests (e.g. tick/simulation)'),
    ]

    def initialize_options(self):
        """Set default values for options."""
        self.start_dir = '.'

    def run(self):
        loader = unittest.TestLoader()
        alltests = loader.discover(self.start_dir, pattern="*_test.py")
        result = unittest.TextTestRunner(verbosity=2).run(alltests)
        sys.exit(not result.wasSuccessful())


class RunTestSuites(TickCommand):
    description = 'run tick Python and C++ tests'

    def run(self):
        self.run_command('cpptest')
        self.run_command('pytest')


class CleanTick(clean):
    description = 'cleans all generated and built files'

    def run(self):
        seconds_until_clean = 5

        print("Cleaning source directories in %d seconds..." %
              seconds_until_clean)

        time.sleep(seconds_until_clean)

        clean.run(self)

        shutil.rmtree(build_dir, ignore_errors=True)

        patterns = [
            '**/*.so',
            '**/*_wrap.cpp',
            '**/__pycache__/*.pyc',
            '**/__pycache__',
        ]

        for paths in (pathlib.Path(os.curdir).glob(p) for p in patterns):
            for path in paths:
                print("Deleting {}".format(path))

                if path.is_dir():
                    path.rmdir()
                else:
                    path.unlink()
from pprint import pprint
pprint(tick_modules)

setup(name="tick",
      version='0.3.0.0',
      author="Emmanuel Bacry, "
             "Stephane Gaiffas, "
             "Martin Bompaire, "
             "Søren V. Poulsen, "
             "Maryan Morel, "
             "Simon Bussy",
      author_email='martin.bompaire@polytechnique.edu, '
                   'soren.poulsen@polytechnique.edu',
      url="https://x-datainitiative.github.io/tick/",
      description="Module for statistical learning, with a particular emphasis "
                  "on time-dependent modelling",
      ext_modules=tick_modules,
      install_requires=['numpy',
                        'scipy',
                        'numpydoc',
                        'matplotlib',
                        'sphinx',
                        'pandas',
                        'scikit-learn'],
      packages=find_packages(),
      cmdclass={'build': TickBuild,
                'install': TickInstall,
                'makecpptest': BuildCPPTests,
                'runcpptest': RunCPPTests,
                'cpptest': BuildRunCPPTests,
                'cpplint': RunCPPLint,
                'pytest': RunPyTests,
                'pylint': RunPyLint,
                'test': RunTestSuites,
                'clean': CleanTick},
      classifiers=['Development Status :: 3 - Alpha',
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Programming Language :: C++',
                   'Programming Language :: Python',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   'License :: OSI Approved :: BSD License'],
      )
