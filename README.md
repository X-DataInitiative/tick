
[![Build Status](https://travis-ci.com/X-DataInitiative/tick.svg?token=sQJ9HAvgQkqM6Z5CF631&branch=dev)](https://travis-ci.com/X-DataInitiative/tick)

# tick

_tick_ is a Python 3 module for statistical learning, with a particular emphasis on time-dependent modelling. It is distributed under the 3-Clause BSD license, see [LICENSE.txt](LICENSE.txt).

The project was started in 2016 by Emmanuel Bacry, Martin Bompaire, Stéphane Gaïffas and Søren Vinther Poulsen at the [Center for Applied Mathematics](http://www.cmap.polytechnique.fr/) of [École Polytechnique](https://www.polytechnique.edu), France.
The list of contributors is available in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Installation

### Requirements

_tick_ requires Python 3.4 or newer. Please have the required Python dependencies in your Python environment:

- numpy
- sciPy
- numpydoc
- sphinx
- scikit-learn
- bokeh
- matplotlib
- pandas

If you build and install _tick_ via _pip_ these dependencies will automatically be resolved.

If not, you can install all of these dependencies easily using:

    pip install -r requirements.txt

### Source installations

For source installations, a C++ compiler capable of compiling C++11 source code (such as recent versions of [gcc](https://gcc.gnu.org/) or [clang](https://clang.llvm.org/)) must be available.
In addition, SWIG version 3.07 or newer must also be available.

### Install using _pip_

_tick_ is available via _pip_. In your local Python environment (or global, with sudo rights), do:

    pip install tick

Installation may take a few minutes to build and link C++ extensions. _tick_ should now be available.

### Manual Install

It's possible to manually build and install tick via the setup.py script. To do both in one step, do:

    python setup.py build install

This will build all required C++ extensions, and install the extensions in your current site-packages directory.

#### In-place (for developers)

If you wish to work with the source code of _tick_ it's convenient to have the extensions installed inside the source directory.
This way you will not have to install the module for each iteration of the code.

Simply do:

    python setup.py build_ext --inplace

This will build all extensions, and install them directly in the source directory. To use the package outside the build directory, the build path should be added to the `PYTHONPATH` environment variable, as such (replace `$PWD` with the full path to the build directory if necessary):

    export PYTHONPATH=$PYTHONPATH:$PWD

## Help and Support

### Documentation

- Documentation is available on [https://x-datainitiative.github.io/tick](https://x-datainitiative.github.io/tick/).

- Documentation can be compiled and used locally by running `make html` from within the `DOC` directory. This needs to have Sphinx installed.

- Several tutorials and code-samples are available in the documentation.
 
### Communication

To reach the developers of _tick_, please join our community channel on Gitter (https://gitter.im/xdata-tick).

If you've found a bug that needs attention, please raise an issue here on Github.
Please try to be as precise in the bug description as possible, such that the developers and other contributors can address the issue efficiently.

### Citation

If you use _tick_ in a scientific publication, we would appreciate citations.

## Developers

We welcome developers and researchers to contribute to the _tick_ package. In order to do so, we ask that you submit pull requests that will be reviewed by the _tick_ team before being merged into the package source.

### Pull Requests

We ask that pull requests meet the following standards:

- PR contains exactly 1 commit that has clear and meaningful description
- All unit tests pass (Python and C++)
- C++ code follows our style guide ([Google style](https://google.github.io/styleguide/cppguide.html))
- If new functionality is added, new unit tests must be included in the pull request

To ease this process, there is a script that automatically compiles and run all the necessary checks:

    ./build_test.sh

This command needs to be run from the package root directory.
To run all checks, [Python unittest](https://docs.python.org/3/library/unittest.html), [gtest](https://github.com/google/googletest) and [cpplint](https://pypi.python.org/pypi/cpplint) needs to be available. On most systems, this will suffice:

    # Install cpplint
    pip install cpplint

    # Install gtest
    git clone https://github.com/google/googletest.git
    (cd googletest && mkdir -p build && cd build && cmake .. && make && make install)

Our continuous integration tool (Travis CI) will also run these checks upon submission of a pull request.
