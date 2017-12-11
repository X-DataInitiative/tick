#!/usr/bin/env bash

set -e -x

shell_session_update() { :; }

brew update
brew install swig
#brew upgrade pyenv

if [ ! -d googletest ] || [ ! -f googletest/CMakeLists.txt ]; then
  git clone https://github.com/google/googletest
  mkdir -p googletest/build
  pushd googletest/build
  cmake .. && make -s && make -s install
  popd
fi

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"

eval "$(pyenv init -)"

env PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install -s ${PYVER}

pyenv local ${PYVER}

python -m pip install --quiet -U pip
python -m pip install --quiet numpy pandas
python -m pip install -r requirements.txt
python -m pip install sphinx pillow
python -m pip install cpplint
pyenv rehash

