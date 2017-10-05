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

[ "$arraylength" == "0" ] && FILES=($(find tick -name "*gtest.cpp"))

cd $CWD/.. 
ROOT=$PWD
source $ROOT/sh/configure_env.sh

[ ! -z "$CXXFLAGS" ] && CARGS="$CXXFLAGS $CARGS"

case "${unameOut}" in
    Linux*)     LDARGS="${LDARGS} $(mkn -G nix_largs)";;
    Darwin*)    LDARGS="${LDARGS} $(mkn -G bsd_largs)";;
    CYGWIN*)    LDARGS="${LDARGS}";;
    MINGW*)     LDARGS="${LDARGS}";;
    *)          LDARGS="${LDARGS}";;
esac
[ ! -z "$LDFLAGS" ] && LDARGS="${LDARGS} ${LDFLAGS}"

cd $CWD/..
export PYTHONPATH=$PWD

mkn build -p gtest -tSa "-fPIC -O2 -DNDEBUG -DGTEST_CREATE_SHARED_LIBRARY" \
    -d google.test,+

for FILE in "${FILES[@]}"; do

    echo FILE $FILE

    mkn clean build run -p gtest -a "${CARGS}" \
        -tl "${LDARGS}" -b "$PY_INCS" \
        -M "${FILE}" -P lib_name=$LIB_POSTFIX \
        -B $B_PATH
        
done
