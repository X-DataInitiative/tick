
[![PyPI version](https://badge.fury.io/py/tick.svg)](https://badge.fury.io/py/tick)
[![Build Status](https://travis-ci.org/X-DataInitiative/tick.svg?branch=master)](https://travis-ci.org/X-DataInitiative/tick)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# tick

_tick_ is a Python 3 module for statistical learning, with a particular emphasis on time-dependent modeling. It is distributed under the 3-Clause BSD license, see [LICENSE.txt](LICENSE.txt).

The project was started in 2016 by Emmanuel Bacry, Martin Bompaire, Stéphane Gaïffas and Søren Vinther Poulsen at the [Datascience initiative](https://portail.polytechnique.edu/datascience/en) of [École Polytechnique](https://www.polytechnique.edu/en), France. The list of contributors is available in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Quick description

_tick_ is a machine learning library for Python 3. The focus is on statistical learning for time dependent systems, such as point processes. Tick features  also tools for generalized linear models and a generic optimization toolbox.
The core of the library is an optimization module providing model computational classes, solvers and proximal operators for regularization. It comes also with inference and simulation tools intended for end-users who for example can easily:

- Perform linear, logistic or Poisson regression
- Simulate point Hawkes processes with standard or exotic kernels. 
- Infer Hawkes models with various assumptions on the kernels: exponential or sum of exponential kernels, linear combination of basis kernels, sparse interactions, etc.

A comprehensive list of examples can be found at

- [https://x-datainitiative.github.io/tick/auto_examples/index.html](https://x-datainitiative.github.io/tick/auto_examples/index.html) 

and the documentation is available at

- [https://x-datainitiative.github.io/tick](https://x-datainitiative.github.io/tick/)

The paper associated to this library has been published at

- [https://arxiv.org/abs/1707.03003](https://arxiv.org/abs/1707.03003)

If you use _tick_ in a scientific publication, we would appreciate citations.

<img align="right" src="https://raw.githubusercontent.com/X-DataInitiative/tick/master/doc/images/intel_logo.png" width="200" alt="intel logo" />
The <i>tick</i> library is released with the support of Intel®. It uses the Intel® Math Kernel Library (MKL) optimized for Intel® Xeon Phi™ and Intel® Xeon™ processors. <i>tick</i> runs efficiently on everything from desktop computers to powerful high-performance servers.


## Use cases

_tick_ is used for many industrial applications including:

* [A joint work](https://portail.polytechnique.edu/datascience/fr/node/329) with the French national social security (CNAMTS) to analyses a huge health-care database, that describes the medical care provided to most of the French citizens. For this project, _tick_ is used to detect weak signals in pharmacovigilance, in order quantify the impact of drugs exposures to the occurrence of adverse events.
   
* High-frequency order book modeling in finance, in order to understand the interactions between different event types and/or between different assets, leveraging the full time resolution available in the original data.

* Analyze the propagation of information in social media. Thanks to a dataset collected during 2017's presidential French election campaign on Twitter, _tick_ is used to recover, for each topic, the network across which information spreads inside the political sphere. 
  

## Installation

### Requirements

_tick_ currently works only on Linux/OSX systems and requires Python 3.4 or newer. Please have the required Python dependencies in your Python environment:

- numpy
- scipy
- numpydoc
- scikit-learn
- matplotlib
- pandas

If you build and install _tick_ via _pip_ these dependencies will automatically be resolved. If not, you can install all of these dependencies easily using:

    pip install -r requirements.txt

[Swig](http://www.swig.org/Doc3.0/SWIGDocumentation.html) might also be necessary if precompiled binaries are not available for your distribution.

### Source installations

For source installations, a C++ compiler capable of compiling C++11 source code (such as recent versions of [gcc](https://gcc.gnu.org/) or [clang](https://clang.llvm.org/)) must be available. In addition, SWIG version 3.07 or newer must also be available.

### Install using _pip_

_tick_ is available via _pip_. In your local Python environment (or global, with sudo rights), do:

    pip install tick

Installation may take a few minutes to build and link C++ extensions. At this point _tick_ should be ready to use available (if necessary, you can add _tick_ to the `PYTHONPATH` as explained below).

### Manual Install

First you need to clone the repository with 

    git clone https://github.com/X-DataInitiative/tick.git

and then initialize its submodules (such as cereal) with
 
    git submodule update --init

It's possible to manually build and install tick via the setup.py script. To do both in one step, do:

    python setup.py build install

This will build all required C++ extensions, and install the extensions in your current site-packages directory.

#### In-place installation (for developers)

If you wish to work with the source code of _tick_ it's convenient to have the extensions installed inside the source directory. This way you will not have to install the module for each iteration of the code. Simply do:

    python setup.py build_ext --inplace

This will build all extensions, and install them directly in the source directory. To use the package outside of the build directory, the build path should be added to the `PYTHONPATH` environment variable (replace `$PWD` with the full path to the build directory if necessary):

    export PYTHONPATH=$PYTHONPATH:$PWD

Note also that special scripts, intended for developers, are available. `./clean_build_test.sh` removes all compiled binaries and runs the full compilation process, with all `C++` and `Python` unit-tests. This can take some time.
Similarly, `./build_test.sh` runs the full compilation process, without removing first binaries, while `full_clean.sh` only removes them.

## Help and Support

### Documentation

Documentation is available on 

- [https://x-datainitiative.github.io/tick](https://x-datainitiative.github.io/tick/)

This documentation is built with `Sphinx` and can be compiled and used locally by running `make html` from within the `doc` directory. This obviously needs to have `Sphinx` installed. Several tutorials and code-samples are available in the documentation.
 
### Communication

To reach the developers of _tick_, please join our community channel on Gitter (https://gitter.im/xdata-tick).

If you've found a bug that needs attention, please raise an issue here on Github. Please try to be as precise in the bug description as possible, such that the developers and other contributors can address the issue efficiently.

### Citation

If you use _tick_ in a scientific publication, we would appreciate citations. You can use the following bibtex entry:

    @ARTICLE{2017arXiv170703003B,
      author = {{Bacry}, E. and {Bompaire}, M. and {Ga{\"i}ffas}, S. and {Poulsen}, S.},
      title = "{tick: a Python library for statistical learning, with 
        a particular emphasis on time-dependent modeling}",
      journal = {ArXiv e-prints},
      eprint = {1707.03003},
      year = 2017,
      month = jul
    }


## Developers

We welcome developers and researchers to contribute to the _tick_ package. In order to do so, we ask that you submit pull requests that will be reviewed by the _tick_ team before being merged into the package source.

### Pull Requests

We ask that pull requests meet the following standards:

- PR contains exactly 1 commit that has clear and meaningful description
- All unit tests pass (Python and C++)
- C++ code follows our style guide ([Google style](https://google.github.io/styleguide/cppguide.html), this is tested if you use provided scripts, such as `./build_test.sh`)
- If new functionality is added, new unit-tests must be included in the pull request

In order to run these tests, you need:

- [Python unittest](https://docs.python.org/3/library/unittest.html)
- [gtest](https://github.com/google/googletest) 
- [cpplint](https://pypi.python.org/pypi/cpplint)

On most systems, this will suffice:

    # Install cpplint
    pip install cpplint

    # Install gtest
    git clone https://github.com/google/googletest.git
    (cd googletest && mkdir -p build && cd build && cmake .. && make && make install)

Our continuous integration tool (Travis CI) will also run these checks upon submission of a pull request.
