#!/usr/bin/env bash

set -e -x

# Copy tick source from host to container
cp -R /io src
cd src

eval "$(pyenv init -)"

pyenv global ${PYVER}
pyenv local ${PYVER}

## disabled for the moment - re-enable later
#python -m pip install yapf --upgrade
#python -m yapf --style tools/code_style/yapf.conf -i tick examples --recursive
# (( $(git diff tick | wc -l) > 0 )) && echo \
# "Python has not been formatted : Please run ./sh/format_python.sh and recommit" \
#   && exit 2

python -m pip install -r requirements.txt & PYPID=$!
if [ ! -d googletest ] || [ ! -f googletest/CMakeLists.txt ]; then
  git clone https://github.com/google/googletest
  mkdir -p googletest/build
  cd googletest/build
  cmake .. && make -s && make -s install
  cd ../..
fi
kill -s 0 $PYPID && wait $PYPID
python setup.py cpplint
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
