#!/usr/bin/env bash

set -e -x

# Copy tick source from host to container
cp -R /io src
cd src
export PATH="/root/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
pyenv global ${PYVER}
pyenv local ${PYVER}

python -m pip install pip --upgrade
python -m pip install -r requirements.txt
python -m pip install cpplint --upgrade
python setup.py cpplint
swig -version # should be 4.0.0
PYMAJ=$(python -c "import sys; print(sys.version_info[0])")
PYMIN=$(python -c "import sys; print(sys.version_info[1])")

python setup.py build_ext --inplace pytest cpptest

export PYTHONPATH=${PYTHONPATH}:`pwd` && (cd doc && make doctest)
