#!/usr/bin/env bash

set -x

shell_session_update() { :; }

eval "$(pyenv init -)"

pyenv global ${PYVER}
PYVNUM="$(echo ${PYVER} | cut -d ' ' -f 2 )"
ROOT=$PWD
cd $ROOT/sh
source configure_var.sh
source build_options.sh
cd $ROOT

CAN_CONFIGURE=0
SHOULD_BUILD=$(should_build_any_CI)
if (( SHOULD_BUILD == 0 )); then
  CAN_CONFIGURE=$(can_configure_bintray_travis $PYVNUM ${BINTRAY_MAC_INFO_URL})
  if (( CAN_CONFIGURE == 0 )); then
  	SHOULD_BUILD=1
  fi
fi
if (( SHOULD_BUILD == 1 )); then
  python setup.py cpplint
  set +e
  python setup.py build_ext -j 2 --inplace # mac builds are slow but -j can fail so run twice
  rm -rf build/lib # force relinking of libraries in case of failure
  set -e
  python setup.py build_ext --inplace cpptest
else
  travis_bintray_configure $PYVNUM ${BINTRAY_MAC_INFO_URL}
fi
python setup.py pytest
if (( SHOULD_BUILD == 1 )); then
  find build -name cpp-test -type d | xargs rm -rf
  find build -name array_test -type d | xargs rm -rf
  python setup.py bdist_wheel
  basename $(ls dist/*.whl) > dist/${PYVNUM}.info
fi

export PYTHONPATH=${PYTHONPATH}:`pwd`
for f in $(find examples -maxdepth 1 -type f -name "*.py"); do
  FILE=$(basename $f)
  FILE="${FILE%.*}"    # skipping it takes too long
  [[ "plot_asynchronous_stochastic_solver" != "$FILE" ]] && \
    DISPLAY="-1" python -c "import tick.base; import examples.$FILE"
done

mkdir -p dist # we give an empty directory if no build
