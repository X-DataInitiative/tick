#!/bin/sh

TICK_DEFAULT_CLONE_FROM=git@github.com:X-DataInitiative/mlpp.git
TICK_CLONE_FROM=${TICK_DEFAULT_CLONE_FROM}

if [ $# -eq 1 ]
  then
    TICK_CLONE_FROM=$1
fi

echo "Cloning from ${TICK_CLONE_FROM}"
git clone ${TICK_CLONE_FROM} tick

CUR_DIR=$(pwd)

(
  cd tick  && MLPP_NO_CPPLINT=true ./build_test.sh

  TICK_BUILD_DIR=${CUR_DIR}/tick

  TICK_LIBRARY_PATH=${TICK_BUILD_DIR}/mlpp/base/array/build:${TICK_BUILD_DIR}/mlpp/base/utils/build:${TICK_BUILD_DIR}/mlpp/base/math/build:${TICK_BUILD_DIR}/mlpp/random/build:${TICK_BUILD_DIR}/mlpp/optim/prox/build:${TICK_BUILD_DIR}/mlpp/optim/model/build

  export PYTHONPATH=${PYTHONPATH}:${TICK_BUILD_DIR}
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${TICK_LIBRARY_PATH}

  cd DOC && make clean html
)

cp -Rv tick/DOC/_build/html/* .

git add -A
git commit -am "Website update"
git push

cd ${CUR_DIR}
