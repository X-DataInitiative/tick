#!/usr/bin/env bash

set -ex

# Copy tick source from host to container
cp -R /io src
cd src
export PATH="/root/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
pyenv global ${PYVER}
pyenv local ${PYVER}

swig -version # should be 4.0.0

PYVNUM="$(echo ${PYVER} | cut -d ' ' -f 2 )"
ROOT=$PWD

python -m pip install pip --upgrade
python -m pip install -r requirements.txt
python -m pip install psutil wheel --upgrade

cd $ROOT/sh
source configure_var.sh
source build_options.sh
cd $ROOT

PYMAJ=$(python -c "import sys; print(sys.version_info[0])")
PYMIN=$(python -c "import sys; print(sys.version_info[1])")

CAN_CONFIGURE=0
SHOULD_BUILD=$(should_build_any_CI)
if (( SHOULD_BUILD == 0 )); then
  CAN_CONFIGURE=$(can_configure_bintray_travis $PYVNUM ${BINTRAY_NIX_INFO_URL})
  if (( CAN_CONFIGURE == 0 )); then
  	SHOULD_BUILD=1
  fi
fi
if (( SHOULD_BUILD == 1 )); then
  python -m pip install cpplint --upgrade
  python setup.py cpplint
  if (( PYMAJ == 3 )) && (( PYMIN == 6 )); then
    set +e
    python setup.py build_ext -j 2 --inplace
    rm -rf build/lib # force relinking of libraries in case of failure
    set -e
  fi
  python setup.py build_ext --inplace cpptest
else
  travis_bintray_configure $PYVNUM ${BINTRAY_NIX_INFO_URL}
fi
python setup.py pytest
if (( SHOULD_BUILD == 1 )); then
  find build -name cpp-test -type d | xargs rm -rf
  find build -name array_test -type d | xargs rm -rf
  if (( PYMAJ == 3 )) && (( PYMIN == 7 )); then
    python -m pip install wheel
  fi
  python setup.py bdist_wheel
  basename $(ls dist/*.whl) > dist/${PYVNUM}.info
fi
mkdir -p dist # we give an empty directory if no build

cp -r dist /io

export PYTHONPATH=${PYTHONPATH}:`pwd` && (cd doc && make doctest)
