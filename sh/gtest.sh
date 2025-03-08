#!/usr/bin/env bash
# Author: Philip Deegan / philip.deegan@gmail.com
set -ex
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd $CWD/../lib
FILES=$(find cpp-test -name "*_gtest.cpp")
cmd(){
  echo "mkn -M $1 -Op gtest"
}
for test_file in ${FILES[@]}; do
  CMD=$(cmd $test_file)
  $CMD clean build
  $CMD run || $CMD run || $CMD run # retry if failed
done
