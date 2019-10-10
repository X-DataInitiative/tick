#!/usr/bin/env bash

set -ex

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $CWD/../..
ROOT=$PWD

python -V
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install psutil wheel

set +e
python setup.py build_ext -j 2 --inplace
rm -rf build/lib # force relinking of libraries in case of failure
set -e
python setup.py build_ext --inplace

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

KLOG=3 $ROOT/sh/gtest.sh

python -m unittest discover -v . "*_test.py"
python setup.py bdist_wheel

rm -rf build
rm -rf lib/bin
