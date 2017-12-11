#!/usr/bin/env bash

set -e -x

shell_session_update() { :; }

eval "$(pyenv init -)"

pyenv global ${PYVER}

python setup.py cpplint build_ext --inplace cpptest pytest

export PYTHONPATH=${PYTHONPATH}:`pwd` && cd doc && make doctest

