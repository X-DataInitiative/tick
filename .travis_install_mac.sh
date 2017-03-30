#!/usr/bin/env bash

brew update
brew install swig pyenv

case "${TOXENV}" in
    py34)
        env PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install -s 3.4.5
        pyenv local 3.4.5
        ;;
    py35)
        env PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install -s 3.5.2
        pyenv local 3.5.2
        ;;
esac

eval "$(pyenv init -)"

if [ ! -d googletest ] || [ ! -f googletest/CMakeLists.txt ]
    then
        git clone https://github.com/google/googletest.git
        (cd googletest && mkdir -p build && cd build && cmake .. && make -s)
fi

export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

(cd googletest && cd build && sudo make -s install)
