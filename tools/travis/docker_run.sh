#!/usr/bin/env bash

set -e -x

# Copy tick source from host to container
cp -R /io src
cd src

( apt-get update && apt-get remove --purge -y swig* && apt-get autoremove && \
  apt-get install -y autotools-dev automake gawk bison flex && \
  git clone https://github.com/swig/swig -b rel-4.0.0 swig && \
  cd swig && ./autogen.sh && ./configure --without-pcre && \
  make && make install ) & SWIG_PID=$!

eval "$(pyenv init -)"
pyenv global ${PYVER}
pyenv local ${PYVER}

## disabled for the moment - re-enable later
#python -m pip install yapf --upgrade
#python -m yapf --style tools/code_style/yapf.conf -i tick examples --recursive
# (( $(git diff tick | wc -l) > 0 )) && echo \
# "Python has not been formatted : Please run ./sh/format_python.sh and recommit" \
#   && exit 2

python -m pip install -r requirements.txt
wait $SWIG_PID

( git clone https://github.com/google/googletest -b master --depth 1 && \
  mkdir -p googletest/build && cd googletest/build && \
  cmake .. && make -s && make -s install) & GTEST_PID=$!

python setup.py cpplint
wait $GTEST_PID
swig -version # should be 4.0.0
PYMAJ=$(python -c "import sys; print(sys.version_info[0])")
PYMIN=$(python -c "import sys; print(sys.version_info[1])")
if (( PYMAJ == 3 )) && (( PYMIN == 6 )); then
  set +e
  python setup.py build_ext -j 2 --inplace
  rm -rf build/lib # force relinking of libraries in case of failure
  set -e
fi
python setup.py build_ext --inplace cpptest pytest

if (( PYMAJ == 3 )) && (( PYMIN == 7 )); then
  echo "Skipping doctest as sphinxext.google_analytics is missing on Travis for 3.7"
else
  export PYTHONPATH=${PYTHONPATH}:`pwd` && (cd doc && make doctest)
fi
