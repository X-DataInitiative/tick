#!/usr/bin/env bash

set -ex

shell_session_update() { :; }

brew update
brew install swig

( git clone https://github.com/google/googletest && \
  mkdir -p googletest/build && cd googletest/build && \
  cmake .. && make -s && make -s install) & GTEST_PID=$!

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
export CC="clang"
export CXX="clang++"

eval "$(pyenv init -)"

env CFLAGS="-I$(xcrun --show-sdk-path)/usr/include" PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install -s ${PYVER}

pyenv local ${PYVER}

PYMAJ=$(python -c "import sys; print(sys.version_info[0])")
PYMIN=$(python -c "import sys; print(sys.version_info[1])")

# It has been observed that python 3.7 works better with ccache, but 3.6 does not
if (( PYMAJ == 3 )) && (( PYMIN == 7 )); then
  brew install ccache
  export PATH="/usr/local/opt/ccache/libexec:$PATH"
  export CC="ccache clang"
  export CXX="ccache clang++"
fi

python -m pip install --quiet -U pip
python -m pip install --quiet numpy pandas
python -m pip install -r requirements.txt
python -m pip install sphinx pillow cpplint
[[ "${PYVER}" != "3.7.0" ]] && python -m pip install tensorflow # does not yet exist on python 3.7
pyenv rehash
wait $GTEST_PID
