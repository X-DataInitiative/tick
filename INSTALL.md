

## Installation

### Requirements

_tick_ currently works on Linux/OSX (Windows is experimental) systems and requires Python 3.6 or newer. Please have the required Python dependencies in your Python environment:

- numpy
- scipy
- numpydoc
- scikit-learn
- matplotlib
- pandas

If you build and install _tick_ via _pip_ these dependencies will automatically be resolved. If not, you can install all of these dependencies easily using:

    pip install -r requirements.txt

[Swig](http://www.swig.org/Doc4.0/SWIGDocumentation.html) might also be necessary if precompiled binaries are not available for your distribution.

### Source installations

For source installations, a C++ compiler capable of compiling C++11 source code (such as recent versions of [gcc](https://gcc.gnu.org/) or [clang](https://clang.llvm.org/)) must be available. In addition, SWIG version 4.0 or newer must also be available. It is also recommended to build swig if necessary without PCRE

The following script is provided as is with not assurance for working, and may require "sudo" permissions:

    git clone https://github.com/swig/swig -b rel-4.0.0 swig && \
    cd swig && ./autogen.sh && ./configure --without-pcre && \
    make && make install

It also may require the following (styled for a Debian type system)

    apt-get install -y autotools-dev automake gawk bison flex

### Install using _pip_

_tick_ is available via _pip_. In your local Python environment (or global, with sudo rights), do:

    pip install tick

Installation may take a few minutes to build and link C++ extensions. At this point _tick_ should be ready to use available (if necessary, you can add _tick_ to the `PYTHONPATH` as explained below).

### Manual Install - Linux -OSX

First you need to clone the repository with

    git clone https://github.com/X-DataInitiative/tick.git

and then initialize its submodules (such as cereal) with

    git submodule update --init

It's possible to manually build and install tick via the setup.py script. To do both in one step, do:

    python setup.py build install

This will build all required C++ extensions, and install the extensions in your current site-packages directory.

### Manual Install on Windows (Experimental)

  Download and install Microsoft Windows Visual Build Tools

    https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017

  Download python and swig, and add their respective installation directories to the PATH environment variable.

    python  3.6   - https://www.python.org/downloads/windows/
    swigwin 4.0.* - http://www.swig.org/download.html

  Download wheel files from http://www.lfd.uci.edu/~gohlke/pythonlibs - this is not required for Anaconda

    numpy-1.13.*+mkl-cp36-cp36m-win_amd64.whl
    numpydoc-0.7.0-py2.py3-none-any.whl
    scipy-0.19.*-cp36-cp36m-win_amd64.whl

  Note: To ascertain which version of the wheel files to download you can run

    python3 -c "import distutils; from distutils import sysconfig; print(sysconfig.get_config_var('SO'))"

  Install the wheel files with pip - this is not required for Anaconda

    pip install wheel
    pip install numpy-1.13.*+mkl-cp36-cp36m-win_amd64.whl
    pip install numpydoc-0.7.0-py2.py3-none-any.whl
    pip install scipy-0.19.*-cp36-cp36m-win_amd64.whl

  Export Visual Studio environment variables for your architecture

    /path/to/VS/VC/Auxiliary/Build/vcvarsall.bat [amd64,x86]

  The default path for "vcvarsall.bat" for Visual Studio Build Tools:

    C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvarsall.bat [amd64,x86]

  Build Tick C++ sources

    python setup.py build_ext --inplace

  Test your build

    SET PYTHONPATH=%cd%
    python setup.py pytest

####  Automated windows installation process (Requires CYGWIN or MSYS)

  Some scripts have been put together to facilitate in the system setup for Windows
  This does also require manually downloading the wheel files.
  This can be used but executing the following file:

    ./tools/windows/install.sh

#### In-place installation (for developers)

If you wish to work with the source code of _tick_ it's convenient to have the extensions installed inside the source directory. This way you will not have to install the module for each iteration of the code. Simply do:

    python setup.py build_ext --inplace

This will build all extensions, and install them directly in the source directory. To use the package outside of the build directory, the build path should be added to the `PYTHONPATH` environment variable (replace `$PWD` with the full path to the build directory if necessary):

    export PYTHONPATH=$PYTHONPATH:$PWD

Note also that special scripts, intended for developers, are available. `./clean_build_test.sh` removes all compiled binaries and runs the full compilation process, with all `C++` and `Python` unit-tests. This can take some time.
Similarly, `./build_test.sh` runs the full compilation process, without removing first binaries, while `full_clean.sh` only removes them.

It is possible to build with support for Intel MKL in the tick C++ libraries, to do so requires having the intel MKL libraries installed and running the build command like so (example provided is for linux with MKL installed to /opt/intel)

    MKLROOT=/opt/intel/mkl python setup.py build_ext --inplace

On MacOS it's possible an error may occur when trying to find a "intel_thread.dyld"
To fix this try:

export DYLD_LIBRARY_PATH="/opt/intel/lib:$DYLD_LIBRARY_PATH"

Replacing "/opt/intel/lib" to "$MKLROOT/../lib" if necessary.

Upgrading to the latest version of MKL may also resolve this.

### Alternative installation method

Also supported is the build tool [maiken](https://github.com/dekken/maiken) which works with various compilers on major platforms.

To use maiken, either download a provided [binary](https://github.com/dekken/maiken/tree/binaries) or compile the source following the instructions in the maiken README @ [Part 3: How to build Maiken]( )

Once "mkn" is installed on your system, you should execture the shell script "./mkn.sh" to compile tick.

As with tick in general, this requires python3 and all python pre-requisites. numpy/scipy/etc

This method is provided mainly for developers to decrease compilation times.

To use [ccache](https://ccache.samba.org/) (or similar) with mkn, edit your ~/.maiken/settings.yaml file so it looks like one [here](https://github.com/Dekken/maiken/wiki/Alternative-configs) - this can greatly decrease repeated compilation times. ccache is a third party tool which may have to be installed manually on your system. ccache does not work on Windows.

On windows it may be required to configure the maiken settings.yaml manually if the environment variables exported by the Visual Studio batch file "vcvars" are not found, example configurations can be found [here](https://github.com/Dekken/maiken/wiki)

Notes:

Logging from mkn can be enabled with the "KLOG" environment variable

    KLOG=1 # INFO
    KLOG=2 # ERROR
    KLOG=3 # DEBUG

If there is a compile error it should be displayed without KLOG being set.


### Python 3.5

Python 3.5 is not officially supported any more, but it should still work.
To build with it, edit "python_min_ver = (3, 6, 0)" in setup.py.

