#!/bin/sh

set -ex

# Copy tick source from host to container
cp -R /io src
cd src

eval "$(pyenv init -)"

pyenv global ${PYVER}

python setup.py cpplint

set +e
python setup.py build_ext --inplace -j4
python setup.py build_ext --inplace -j4 
python setup.py build_ext --inplace -j4 
python setup.py build_ext --inplace -j4 

set -e

python setup.py build_ext --inplace -j4 
python setup.py cpptest pytest

export PYTHONPATH=${PYTHONPATH}:`pwd` && (cd doc && make doctest)
