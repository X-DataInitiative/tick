#!/usr/bin/env bash
#
# Author: Philip Deegan
# Email : philip.deegan@polytechnique.edu 
# Date  : 26 - September - 2017
#
# This script builds and launches google test files for tick
#
# Requires the mkn build tool
#
# Input arguments are optional but if used are to be the 
#  test files to be compiled and run, otherwise all files
#  with the syntax "*gtest.cpp" are used
#
# Tests are expected to finish with the following code
#
# #ifdef ADD_MAIN
# int main(int argc, char** argv) {
#   ::testing::InitGoogleTest(&argc, argv);
#   ::testing::FLAGS_gtest_death_test_style = "fast";
#   return RUN_ALL_TESTS();
# }
# #endif  // ADD_MAIN
#
# This is used to inject main functions for testing/execution
# 
######################################################################

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

FILES=()
array=( "$@" )
arraylength=${#array[@]}
for (( i=1; i<${arraylength}+1; i++ ));
do
   FILES+=("${array[$i-1]}")
done

pushd $CWD/.. 2>&1 > /dev/null
ROOT=$PWD
popd 2>&1 > /dev/null

source $ROOT/sh/configure_env.sh

pushd $ROOT/lib 2>&1 > /dev/null
  [ "$arraylength" == "0" ] && FILES=($(find src/cpp-test -name "*gtest.cpp"))

  [ ! -z "$CXXFLAGS" ] && CARGS="$CXXFLAGS $CARGS"
  [ ! -z "$LDFLAGS" ] && LDARGS="${LDARGS} ${LDFLAGS}"

  mkn build -p gtest -tSa "-fPIC -O2 -DNDEBUG -DGTEST_CREATE_SHARED_LIBRARY" \
      -d google.test,+ ${MKN_X_FILE[@]} "${MKN_WITH[@]}"

  LIB=""
  if [[ "$unameOut" == "Darwin"* ]]; then
    for P in "${PROFILES[@]}"; do
      LIB="${LIB} -Wl,-rpath,@loader_path/../../$(dirname ${LIBRARIES[$(hash_index $P)]})"
    done
  fi

  RUN="run"
  [[ $DEBUG == 1 ]] && RUN="dbg"
  set -x
  for FILE in "${FILES[@]}"; do

      echo FILE $FILE

      mkn clean build -p gtest -a "${CARGS}" \
          -tl "${LDARGS} ${LIB}" -b "$PY_INCS" \
          -M "${FILE}" -P "${MKN_P}" "${MKN_WITH[@]}" \
          -B $B_PATH ${RUN} ${MKN_X_FILE[@]}
          
  done

popd 2>&1 > /dev/null