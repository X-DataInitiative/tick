#!/usr/bin/env bash

set -ex

# Copy tick source from host to container
cp -R /io src
cd src

eval "$(pyenv init -)"

pyenv global ${PYVER}
pyenv local ${PYVER}
PYVNUM="$(echo ${PYVER} | cut -d ' ' -f 2 )"
ROOT=$PWD
cd $ROOT/sh
source configure_var.sh
source build_options.sh
cd $ROOT

python -m pip install -r requirements.txt

SHOULD_BUILD=$(should_build_any_CI)
if (( SHOULD_BUILD == 1 )); then
  python setup.py cpplint build_ext --inplace cpptest
else
  nix_travis_bintray_configure $PYVNUM
fi
python setup.py pytest
if (( SHOULD_BUILD == 1 )); then
  find build -name cpp-test -type d | xargs rm -rf
  find build -name array_test -type d | xargs rm -rf
  python setup.py bdist_wheel
  basename $(ls dist/*.whl) > dist/${PYVNUM}.info
fi
mkdir -p dist # we give an empty directory if no build

cp -r dist /io
export PYTHONPATH=${PYTHONPATH}:`pwd` && (cd doc && make doctest)

set +x # travis stuff goes a bit nuts after this
