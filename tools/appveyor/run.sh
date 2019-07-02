#!/usr/bin/env bash

set -ex

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $CWD/../..
ROOT=$PWD

/c/Python37-x64/python -m pip install --upgrade pip

cat appveyor/pip/numpy/numpy-1.16.4+mkl-cp37* > appveyor/pip/numpy-1.16.4+mkl-cp37-cp37m-win_amd64.whl
/c/Python37-x64/Scripts/pip install appveyor/pip/numpy-1.16.4+mkl-cp37-cp37m-win_amd64.whl
/c/Python37-x64/Scripts/pip install appveyor/pip/numpydoc-0.8.0-py2.py3-none-any.whl
/c/Python37-x64/Scripts/pip install appveyor/pip/scipy-1.3.0-cp37-cp37m-win_amd64.whl
/c/Python37-x64/Scripts/pip install -r requirements.txt
/c/Python37-x64/Scripts/pip install tensorflow

set +e
/c/Python37-x64/python setup.py build_ext -j 2 --inplace
rm -rf build/lib # force relinking of libraries in case of failure
set -e
/c/Python37-x64/python setup.py build_ext --inplace

B_OS=$(basename $(find build -maxdepth 1 -name "lib*"))
pushd $ROOT/build/$B_OS/tick
for d in $(find . -type d -name build); do
  cp -r $d/* $ROOT/tick/$d
done
popd

ls -l $ROOT/tick/survival/build

export MKN_CL_PREFERRED=1 # forces mkn to use cl even if gcc/clang are found
# export MKN_COMPILE_THREADS=1 # mkn use 1 thread heap space issue
export SWIG=0 # disables swig for mkn
export CXXFLAGS="-EHsc"

$ROOT/sh/gtest.sh

/c/Python37-x64/python -m unittest discover -v . "*_test.py"
/c/Python37-x64/python setup.py bdist_wheel

rm -rf build
rm -rf lib/bin
