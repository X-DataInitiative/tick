#!/usr/bin/env bash

set -ex

shell_session_update() { :; }

eval "$(pyenv init -)"

pyenv global ${PYVER}
PYVNUM="$(echo ${PYVER} | cut -d ' ' -f 2 )"
ROOT=$PWD
cd $ROOT/sh
source configure_var.sh
source build_options.sh
cd $ROOT

SHOULD_BUILD=$(should_build_any_CI)
if (( SHOULD_BUILD == 1 )); then
  python setup.py cpplint build_ext --inplace cpptest
else
  mac_travis_bintray_configure $PYVNUM
fi
python setup.py pytest
if (( SHOULD_BUILD == 1 )); then
  find build -name cpp-test -type d | xargs rm -rf
  find build -name array_test -type d | xargs rm -rf
  python setup.py bdist_wheel
  basename $(ls dist/*.whl) > dist/${PYVNUM}.info
fi
mkdir -p dist # we give an empty directory if no build

export PYTHONPATH=${PYTHONPATH}:`pwd` && cd doc && make doctest
cd ..
set +x # travis stuff goes a bit nuts after this
