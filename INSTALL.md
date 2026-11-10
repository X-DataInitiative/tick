## Installation

### Install using _pip_

Pre-built wheels are published on PyPI for Linux (x86_64), macOS (x86_64 and arm64) and Windows (x64), for Python 3.11 to 3.14. In your Python environment (a virtualenv is recommended), run:

    pip install tick

Runtime dependencies (numpy, scipy, pandas, scikit-learn, dill, packaging, matplotlib) are resolved automatically by _pip_.

### Installing from source

#### Requirements

- Python 3.11 or newer
- A C++ compiler supporting C++17 (GCC, Clang, or MSVC)
- CMake 3.24 or newer (installed automatically by the build backend if missing)

The build uses [scikit-build-core](https://scikit-build-core.readthedocs.io) and [pybind11](https://pybind11.readthedocs.io); both are fetched automatically as build dependencies.

#### Linux and macOS

Clone the repository and initialize its submodules (such as cereal):

    git clone https://github.com/X-DataInitiative/tick.git
    cd tick
    git submodule update --init

Then build and install from the checkout:

    python -m pip install .

This compiles all C++ extensions and installs _tick_ into the current environment.

#### Windows

Install the [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) with the "Desktop development with C++" workload, then build from a "x64 Native Tools Command Prompt" (or after running `vcvarsall.bat amd64`), so that the MSVC environment is available:

    git clone https://github.com/X-DataInitiative/tick.git
    cd tick
    git submodule update --init
    python -m pip install .

### Development setup

To work on the source code, install _tick_ in editable mode with the development extras:

    python -m pip install -e ".[dev]"

and run the test suite with:

    python -m pytest -q

Helper scripts are available in the `sh` directory:

- `sh/build_test.sh` installs in editable mode with the `dev` extras and runs the Python test suite.
- `sh/full_clean.sh` removes build artifacts.
- `sh/clean_build_test.sh` runs `full_clean.sh` followed by `build_test.sh`.

These scripts use the Python executable given by the `TICK_PYTHON` environment variable, if set.

To build the documentation, install the `docs` extras (`python -m pip install -e ".[docs]"`) and run `make html` from the `doc` directory.
