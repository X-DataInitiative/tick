#!/usr/bin/env bash

set -e

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd $CWD/../.. 2>&1 > /dev/null
ROOT=$PWD
popd 2>&1 > /dev/null

source $ROOT/sh/configure_var.sh
source $ROOT/sh/build_options.sh

PYVNUM=$1

pushd $ROOT 2>&1 > /dev/null

CAN_CONFIGURE=0
SHOULD_BUILD=$(should_build_any_CI)
if (( SHOULD_BUILD == 0 )); then
  CAN_CONFIGURE=$(can_appveyor_configure $PYVNUM ${APPVEYOR_BASE_BIN_URL})
  if (( CAN_CONFIGURE == 0 )); then
  	SHOULD_BUILD=1
  fi
fi
if (( SHOULD_BUILD == 1 )); then
  find build -name cpp-test -type d | xargs rm -rf
  echo 1
else
  appveyor_configure $PYVNUM ${APPVEYOR_BASE_BIN_URL}
  echo 0
fi

popd 2>&1 > /dev/null

exit 0
