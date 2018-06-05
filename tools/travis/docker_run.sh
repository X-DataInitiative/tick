#!/bin/sh

set -e -x

# Copy tick source from host to container
cp -R /io src
cd src

eval "$(pyenv init -)"

pyenv global ${PYVER}
pyenv local ${PYVER}

python -m pip install yapf --upgrade
python -m yapf --style tools/code_style/yapf.conf -i tick --recursive

(( $(git diff tick | wc -l) > 0 )) && echo \
"Python has not been formatted : Please run ./sh/format_python.sh and recommit" \
  && exit 2

python -m pip install -r requirements.txt

python setup.py cpplint build_ext --inplace cpptest pytest

export PYTHONPATH=${PYTHONPATH}:`pwd` && (cd doc && make doctest)
