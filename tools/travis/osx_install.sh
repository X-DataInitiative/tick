#!/usr/bin/env bash

set -e -x

shell_session_update() { :; }

brew update
brew upgrade pyenv
brew install swig


if [ ! -d googletest ] || [ ! -f googletest/CMakeLists.txt ]; then
  git clone https://github.com/google/googletest
  mkdir -p googletest/build
  pushd googletest/build
  cmake .. && make -s && make -s install
  popd
fi

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
export CC="clang"
export CXX="clang++"

eval "$(pyenv init -)"

env PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install -s ${PYVER}

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
python -m pip install sphinx pillow
python -m pip install cpplint
[[ "${PYVER}" != "3.7.0" ]] && python -m pip install tensorflow # does not yet exist on python 3.7

# Check if not already installed
if brew ls --versions geos > /dev/null; then
  echo "geos is already installed"
else
  brew install geos
fi
# needed for basemap see https://github.com/matplotlib/basemap/issues/414#issuecomment-436792915
# python -m pip install https://github.com/jswhit/pyproj/archive/v1.9.5.1rel.zip
python -m pip install https://github.com/matplotlib/basemap/archive/v1.2.0rel.tar.gz
pyenv rehash

