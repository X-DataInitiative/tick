#!/bin/sh

TICK_DEFAULT_CLONE_FROM=git@github.com:X-DataInitiative/tick.git
TICK_CLONE_FROM=${TICK_DEFAULT_CLONE_FROM}

if [ $# -eq 1 ]
  then
    TICK_CLONE_FROM=$1
fi

rm -rf tick
echo "Cloning from ${TICK_CLONE_FROM}"
git clone --recursive ${TICK_CLONE_FROM} tick

CUR_DIR=$(pwd)

(
  cd tick  && TICK_NO_CPPLINT=true ./sh/build_test.sh
  TICK_BUILD_DIR=${CUR_DIR}/tick
  cd DOC && make clean html
)

cp -Rv tick/DOC/_build/html/* .

git add -A
git commit -am "Website update"

cd ${CUR_DIR}
