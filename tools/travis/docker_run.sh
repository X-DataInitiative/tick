#!/bin/sh

set -e -x

# Copy tick source from host to container
cp -R /io src
cd src

eval "$(pyenv init -)"

pyenv global ${PYVER}
pyenv local ${PYVER}

python -m pip install tensorflow
python setup.py cpplint build_ext --inplace cpptest pytest

export PYTHONPATH=${PYTHONPATH}:`pwd` && (cd doc && make doctest)
