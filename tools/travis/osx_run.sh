#!/usr/bin/env bash

set -e -x

shell_session_update() { :; }

eval "$(pyenv init -)"

pyenv global ${PYVER}

python -m pip install yapf --upgrade
python -m yapf --style tools/python/yapf.conf -i tick --recursive

(( $(git diff tick | wc -l) > 0 )) && \
echo "Python has not been formatted : Please run ./sh/format_python.sh and recommit" \
  && exit 2

python setup.py cpplint build_ext --inplace cpptest pytest

export PYTHONPATH=${PYTHONPATH}:`pwd` && cd doc && make doctest

