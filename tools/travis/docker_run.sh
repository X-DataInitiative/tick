#!/usr/bin/env bash

set -e -x

# Copy tick source from host to container
cp -R /io src
cd src

eval "$(pyenv init -)"

pyenv global ${PYVER}
pyenv local ${PYVER}

python -m pip install yapf --upgrade
python -m yapf --style tools/code_style/yapf.conf -i tick examples --recursive

## disabled for the moment - re-enable later
# (( $(git diff tick | wc -l) > 0 )) && echo \
# "Python has not been formatted : Please run ./sh/format_python.sh and recommit" \
#   && exit 2

python -m pip install -r requirements.txt

python setup.py cpplint
set +e
python setup.py build_ext --inplace
rm -rf build/lib # force relinking of libraries in case of failure
set -e
python setup.py build_ext --inplace cpptest pytest

PYMAJ=$(python -c "import sys; print(sys.version_info[0])")
PYMIN=$(python -c "import sys; print(sys.version_info[1])")

if (( PYMAJ == 3 )) && (( PYMIN == 7 )); then
  echo "Skipping doctest as sphinxext.google_analytics is missing on Travis for 3.7"
else
  export PYTHONPATH=${PYTHONPATH}:`pwd` && (cd doc && make doctest)
fi
