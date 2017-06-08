#!/usr/bin/env bash

brew update
brew install swig pyenv
brew upgrade pyenv

if [ ! -d googletest ] || [ ! -f googletest/CMakeLists.txt ]
    then
        git clone https://github.com/google/googletest.git
        (cd googletest && mkdir -p build && cd build && cmake .. && make -s)
fi

(cd googletest && cd build && sudo make -s install)

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"

eval "$(pyenv init -)"

env PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install -s ${PYVER}