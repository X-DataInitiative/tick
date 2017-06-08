#!/usr/bin/env python
# -*- coding: utf8 -*-

# python setup.py build_ext --inplace

"""
setup.py file
"""

import os
import sys
import sysconfig
import platform
from distutils.command.build import build
from setuptools import find_packages, setup
from setuptools.extension import Extension
from setuptools.command.install import install

import numpy as np
from numpy.distutils.system_info import get_info
from scipy.sparse import csr_matrix

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

# We need to understand what kind of ints are used by the sparse matrices of
# scipy
sparsearray = csr_matrix(
    (np.array([1.8, 2, 3, 4]),
     np.array([3, 5, 7, 4]),
     np.array([0, 3, 4])))

sparsearray_type = sparsearray.indices.dtype
if sparsearray_type == np.int32:
    sparse_indices_flag = "-D_SPARSE_INDICES_INT32"
elif sparsearray_type == np.int64:
    sparse_indices_flag = "-D_SPARSE_INDICES_INT64"

if os.name == 'posix':
    if platform.system() == 'Darwin':
        os_version = platform.mac_ver()[0]
        # keep only major + minor
        os_version = '.'.join(os_version.split('.')[:2])
        from distutils.version import LooseVersion

        if LooseVersion(os_version) < LooseVersion('10.9'):
            raise ValueError(
                'You need to have at least mac os 10.9 to build this package')

        # We set this variable manually because anaconda set it to a deprecated
        # one
        os.environ['MACOSX_DEPLOYMENT_TARGET'] = os_version

# How do we create shared libs? Dynamic or bundle?
create_bundle = 'bundle' in sysconfig.get_config_var("LDCXXSHARED")

# Obtain the numpy include directory.
# This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

# Determine if we have an available BLAS implementation
blas_info = get_info("blas_opt", 0)

class SwigExtension(Extension):
    """This only adds information about extension construction, useful for
    library sharing
    """

    def __init__(self, *args, module_ref=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.module_ref = module_ref


class SwigPath:
    """Small class to handle module creation and check project structure
    """

    def __init__(self, module_path, extension_name):
        module_path = os.path.normpath(module_path)

        # Module C++ source directory (e.g. tick/base/array/src)
        self.src = os.path.join(module_path, 'src')

        # Module SWIG interface files directory (e.g. tick/base/array/swig)
        self.swig = os.path.join(module_path, 'swig')

        # Module build directory. Will contain generated .py files, and .so
        # files if built with flag --inplace.
        #
        # E.g. tick/base/array/build
        self.build = os.path.join(module_path, 'build')

        self.extension_name = extension_name
        self.private_extension_name = '_' + extension_name

        # Transform folder path to module path
        self.extension_path = self.build\
                                  .replace('.', '')\
                                  .replace('/', '.')\
                            + '.' + self.private_extension_name

        self._check_configuration()

        # Directory containing built .so files before they are moved either
        # in source (with build flag --inplace) or to site-packages (by install)
        #
        # E.g. build/lib.macosx-10.11-x86_64-3.4
        self.build_dir = "build/lib.{}-{}.{}".format(sysconfig.get_platform(),
                                                     *sys.version_info[:2])

        # Filename of the produced .so file (e.g. _array.so)
        self.lib_filename = '{}{}'.format(self.private_extension_name,
                                          sysconfig.get_config_var('SO'))

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

        # Check that src and swig folders do exists
        for directory in [self.src, self.swig]:
            if not os.path.exists(directory):
                raise FileNotFoundError(exception_base_msg + (
                    exception_missing_directory_msg % directory))
            elif not os.path.isdir(directory):
                raise NotADirectoryError(exception_base_msg + (
                    exception_file_instead_directory_msg % directory))

        # Check that build is a directory (not a file) or create it
        if not os.path.exists(self.build):
            os.mkdir(self.build)
        elif not os.path.isdir(self.build):
            raise NotADirectoryError(exception_base_msg + (
                exception_file_instead_directory_msg % self.build))


def create_extension(extension_name, module_dir,
                     cpp_files, h_files, swig_files, folders=[],
                     include_modules=None, extra_compile_args=None,
                     swig_opts=None):
    swig_path = SwigPath(module_dir, extension_name)
    extension_path = swig_path.extension_path

    # Add directory to filenames
    def add_dir_name(dir_name, filenames):
        return list(os.path.join(dir_name, filename) for filename in filenames)

    cpp_files = add_dir_name(swig_path.src, cpp_files)
    h_files = add_dir_name(swig_path.src, h_files)
    swig_files = add_dir_name(swig_path.swig, swig_files)

    for folder in folders:
        folder_path = os.path.join(swig_path.src, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(file)
                if ext == '.h':
                    h_files += [os.path.join(folder_path, file)]
                elif ext == '.cpp':
                    cpp_files += [os.path.join(folder_path, file)]
                else:
                    print('WARNING: included file %s in folder %s has an '
                          'unknown extension "%s"' % (file, folder, ext))

    min_swig_opts = ['-py3',
                     '-c++',
                     '-modern',
                     '-new_repr',
                     '-I./tick/base/swig/',
                     '-I./tick/base/src/',
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
                              '-I./tick/base/src/',
                              sparse_indices_flag,
                              '-std=c++11',
                              ]

    if extra_compile_args is None:
        extra_compile_args = min_extra_compile_args
    else:
        extra_compile_args.extend(min_extra_compile_args)

    # Added -Wall to get all warnings and -Werror to treat them as errors
    extra_compile_args.append("-Wall")
    extra_compile_args.append("-Werror")

    # This warning is turned off because SWIG generates files that triggers the
    # warning
    extra_compile_args.append("-Wno-uninitialized")

    # Include directory of module
    mod = SwigPath(module_dir, extension_name)
    for opts in [swig_opts, extra_compile_args]:
        opts.extend(["-I%s/" % mod.src, "-I%s/" % mod.swig])

    libraries = []
    library_dirs = []
    runtime_library_dirs = []
    extra_link_args = []
    define_macros = []
    extra_include_dirs = []

    # Deal with (optional) BLAS
    extra_compile_args.extend(blas_info.get("extra_compile_args", []))
    extra_link_args.extend(blas_info.get("extra_link_args", []))
    libraries.extend(blas_info.get("libraries", []))
    library_dirs.extend(blas_info.get("library_dirs", []))
    define_macros.extend(blas_info.get("define_macros", []))

    if any(key == 'HAVE_CBLAS' for key, _ in blas_info['define_macros']):
        define_macros.append(('TICK_CBLAS_AVAILABLE', None))

    if include_modules is None:
        include_modules = []

    # Include all what need for module link
    for mod in include_modules:
        if mod.__class__ != SwigPath:
            raise ValueError("Unhandled class for included module")

        for opts in [swig_opts, extra_compile_args]:
            opts.extend(["-I%s/" % mod.src, "-I%s/" % mod.swig])

        # Add compile-time location of the dependency library
        library_dirs.append(os.path.join(mod.build_dir, mod.build))

        # Because setuptools produces shared object files with non-standard
        # names (i.e. not lib<name>.so) we prepend with colon to specify the
        # name directly
        libraries.append(":" + mod.lib_filename)

        # Make sure that the runtime linker can find shared object dependencies
        # by using the relative path to the dependency library. $ORIGIN refers
        # to the location of the current shared object file at runtime
        runtime_library_dirs.append(
            "\$ORIGIN/%s" % os.path.relpath(mod.build, swig_path.build))

    if platform.system() == 'Linux':
        # On some systems we need to manually define the SONAME such that only
        # the filename is used to resolve the library location at runtime
        # (together with rpath)
        extra_link_args.append('-Wl,-soname,%s' % swig_path.lib_filename)

    for df in debug_flags:
        full_flag = "-D" + df

        extra_compile_args.append(full_flag)

        if df == 'DEBUG_COSTLY_THROW':
            swig_opts.append(full_flag)

    # Adding Cereal serialization library
    extra_include_dirs.append("tick/third_party/cereal/include")

    # Adding numpy include directory
    extra_include_dirs.append(numpy_include)

    if create_bundle:
        core_module = SwigExtension(extension_path, module_ref=swig_path,
                                    sources=swig_files + cpp_files,
                                    extra_compile_args=extra_compile_args,
                                    extra_link_args=extra_link_args,
                                    define_macros=define_macros,
                                    swig_opts=swig_opts,
                                    include_dirs=extra_include_dirs,
                                    depends=h_files,
                                    language="c++", )
    else:
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
                                    language="c++", )

    return core_module


##############################
# Create extensions
##############################

array_extension_info = {
    "cpp_files": ["alloc.cpp"],
    "h_files": ["basearray.h", "abstractarray1d2d.h", "basearray2d.h",
                "alloc.h", "array.h", "array2d.h",
                "sbasearray.h", "sbasearray2d.h", "sarray.h", "sarray2d.h",
                "sparsearray.h", "sparsearray2d.h",
                "sparsearray2d.h", "ssparsearray.h", "ssparsearray2d.h",
                "varray.h", "view.h", "view2d.h",
                "vector_operations.h"],
    "swig_files": ["array_module.i"],
    "module_dir": "./tick/base/array/",
    "extension_name": "array"
}

array_extension = create_extension(**array_extension_info)

base_extension_info = {
    "cpp_files": ["time_func.cpp",
                  "interruption.cpp",
                  "exceptions_test.cpp",
                  "math/t2exp.cpp",
                  "math/normal_distribution.cpp",
                  ],
    "h_files": ["base.h",
                "base_test.h",
                "debug.h",
                "defs.h",
                "serialization.h",
                "time_func.h",

                "parallel/parallel.h",
                "parallel/parallel_utils.h",

                "interruption.h",
                "base_test.h",

                "exceptions_test.h",

                "math/t2exp.h",
                "math/t2exp.inl",
                "math/normal_distribution.h",
                ],
    "swig_files": ["base_module.i"],
    "module_dir": "./tick/base",
    "extension_name": "base",
    "include_modules": [array_extension.module_ref]
}

base_extension = create_extension(**base_extension_info)

base_array_modules = [array_extension.module_ref, base_extension.module_ref]

# TODO: change name of extension to array_test
array_test_extension_info = {
    "cpp_files": ["array_test.cpp", "typemap_test.cpp", "varraycontainer.cpp",
                  "sbasearray_container.cpp", "performance_test.cpp"],
    "h_files": ["array_test.h", "typemap_test.h", "varraycontainer.h",
                "sbasearray_container.h", "performance_test.h"],
    "swig_files": ["array_test.i"],
    "module_dir": "./tick/base/array_test/",
    "extension_name": "array_test",
    "include_modules": base_array_modules,
}

test_extension = create_extension(**array_test_extension_info)

random_extension_info = {
    "cpp_files": ["rand.cpp", "test_rand.cpp"],
    "h_files": ["rand.h", "test_rand.h"],
    "swig_files": ["crandom.i"],
    "module_dir": "./tick/random/",
    "extension_name": "crandom",
    "include_modules": base_array_modules
}

random_extension = create_extension(**random_extension_info)

simulation_extension_info = {
    "cpp_files": ["pp.cpp", "poisson.cpp", "inhomogeneous_poisson.cpp",
                  "hawkes.cpp"],
    "h_files": ["pp.h", "poisson.h", "inhomogeneous_poisson.h",
                "hawkes.h"],
    "folders": ["hawkes_baselines", "hawkes_kernels"],
    "swig_files": ["simulation_module.i"],
    "module_dir": "./tick/simulation/",
    "extension_name": "simulation",
    "include_modules": base_array_modules + [random_extension.module_ref]
}

simulation_extension = create_extension(**simulation_extension_info)

model_core_info = {
    "cpp_files": ["model_labels_features.cpp",
                  "model_generalized_linear.cpp",
                  "model_generalized_linear_with_intercepts.cpp",
                  "model_lipschitz.cpp",
                  "hawkes_fixed_expkern_loglik.cpp",
                  "hawkes_fixed_expkern_leastsq.cpp",
                  "hawkes_fixed_sumexpkern_leastsq.cpp",
                  "hawkes_utils.cpp",
                  "linreg.cpp",
                  "linreg_with_intercepts.cpp",
                  "logreg.cpp",
                  "poisreg.cpp",
                  "coxreg_partial_lik.cpp"],
    "h_files": ["model.h",
                "model_labels_features.h",
                "model_generalized_linear.h",
                "model_generalized_linear_with_intercepts.h",
                "model_lipschitz.h",
                "hawkes_fixed_expkern_loglik.h",
                "hawkes_fixed_expkern_leastsq.h",
                "hawkes_fixed_sumexpkern_leastsq.h",
                "hawkes_utils.h",
                "linreg.h",
                "linreg_with_intercepts.h",
                "logreg.h",
                "poisreg.h",
                "coxreg_partial_lik.h"],
    "folders": ["variants", "base"],
    "swig_files": ["model_module.i"],
    "module_dir": "./tick/optim/model/",
    "extension_name": "model",
    "include_modules": base_array_modules
}

model_core = create_extension(**model_core_info)

prox_core_info = {
    "cpp_files": ["prox.cpp",
                  "prox_separable.cpp",
                  "prox_zero.cpp",
                  "prox_positive.cpp",
                  "prox_l2sq.cpp",
                  "prox_l1.cpp",
                  "prox_l1w.cpp",
                  "prox_tv.cpp",
                  "prox_elasticnet.cpp",
                  "prox_sorted_l1.cpp",
                  "prox_multi.cpp",
                  "prox_equality.cpp",
                  "prox_slope.cpp"],
    "h_files": ["prox.h",
                "prox_separable.h",
                "prox_zero.h",
                "prox_positive.h",
                "prox_l2sq.h",
                "prox_l1.h",
                "prox_l1w.h",
                "prox_tv.h",
                "prox_elasticnet.h",
                "prox_sorted_l1.h",
                "prox_multi.h",
                "prox_equality.h",
                "prox_slope.h"],
    "swig_files": ["prox_module.i"],
    "module_dir": "./tick/optim/prox/",
    "extension_name": "prox",
    "include_modules": base_array_modules
}

prox_core = create_extension(**prox_core_info)

solver_core_info = {
    "cpp_files": ["sto_solver.cpp",
                  "sgd.cpp",
                  "svrg.cpp",
                  "sdca.cpp",
                  "adagrad.cpp"],
    "h_files": ["sto_solver.h",
                "sgd.h",
                "svrg.h",
                "sdca.h",
                "adagrad.h",
                "sto_solver.h"],
    "swig_files": ["solver_module.i"],
    "module_dir": "./tick/optim/solver/",
    "extension_name": "solver",
    "include_modules": base_array_modules + [random_extension.module_ref,
                                             model_core.module_ref,
                                             prox_core.module_ref]
}

solver_core = create_extension(**solver_core_info)

preprocessing_core_info = {
    "cpp_files": ["sparse_longitudinal_features_product.cpp",
                  "longitudinal_features_lagger.cpp"],
    "h_files": ["sparse_longitudinal_features_product.h",
                "longitudinal_features_lagger.h"],
    "swig_files": ["preprocessing_module.i"],
    "module_dir": "./tick/preprocessing/",
    "extension_name": "preprocessing",
    "include_modules": base_array_modules
}

preprocessing_core = create_extension(**preprocessing_core_info)

inference_extension_info = {
    "cpp_files": ["hawkes_conditional_law.cpp", "hawkes_em.cpp",
                  "hawkes_adm4.cpp", "hawkes_basis_kernels.cpp",
                  "hawkes_sumgaussians.cpp"],
    "h_files": ["hawkes_conditional_law.h", "hawkes_em.h",
                "hawkes_adm4.h", "hawkes_basis_kernels.h",
                "hawkes_sumgaussians.h"],
    "swig_files": ["inference_module.i"],
    "module_dir": "./tick/inference/",
    "extension_name": "inference",
    "include_modules": base_array_modules + [model_core.module_ref]
}

inference_extension = create_extension(**inference_extension_info)


class CustomBuild(build):
    def run(self):
        self.run_command('build_ext')
        build.run(self)


class CustomInstall(install):
    def run(self):
        self.run_command('build_ext')
        self.do_egg_install()


setup(name="tick",
      version='0.1.2002',
      author="Emmanuel Bacry, "
             "Stephane Gaiffas, "
             "Martin Bompaire, "
             "Søren Vinther Poulsen, "
             "Maryan Morel, "
             "Simon Bussy",
      author_email='martin.bompaire@polytechnique.edu, '
                   'soren.poulsen@polytechnique.edu',
      url="https://x-datainitiative.github.io/tick/",
      description="Module for statistical learning, with a particular emphasis "
                  "on time-dependent modelling",
      ext_modules=[array_extension, base_extension,
                   test_extension, random_extension, simulation_extension,
                   model_core, prox_core, solver_core, preprocessing_core,
                   inference_extension],
      install_requires=['numpy',
                        'numpydoc',
                        'scipy',
                        'matplotlib',
                        'sphinx',
                        'pandas',
                        'scikit-learn'],
      packages=find_packages(),
      cmdclass={'build': CustomBuild, 'install': CustomInstall},
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


