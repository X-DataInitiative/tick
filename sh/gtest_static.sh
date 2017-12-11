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

cd $CWD/.. 
ROOT=$PWD
source $ROOT/sh/configure_env.sh

cd $ROOT/lib
[ "$arraylength" == "0" ] && FILES=($(find src/cpp-test -name "*gtest.cpp"))

[ ! -z "$CXXFLAGS" ] && CARGS="$CXXFLAGS $CARGS"
[ ! -z "$LDFLAGS" ] && LDARGS="${LDARGS} ${LDFLAGS}"

mkn clean build -p gtest -tKa "-fPIC -O2 -DNDEBUG -DGTEST_CREATE_SHARED_LIBRARY" \
    -d google.test,+ ${MKN_X_FILE[@]}

mkdir -p $ROOT/gtest

for FILE in "${FILES[@]}"; do

    echo FILE $FILE
    rm -rf bin/gtest*

    mkn compile -p gtest -a "${CARGS}" \
        -tb "$PY_INCS" \
        -M "${FILE}" -P "${MKN_P[@]}" \
        ${MKN_X_FILE[@]} -B "$B_PATH" ${MKN_WITH[@]}        

    mv bin/gtest bin/gtest_nodep
    
    for P in "${PROFILES[@]}"; do
        cp bin/$P/obj/* bin/gtest_nodep/obj
    done

    mkn link -p gtest_nodep  \
            -tl "${LDARGS}" \
            -P "${MKN_P}" \
            -KB $B_PATH ${MKN_X_FILE[@]}
    mv bin/gtest_nodep/tick $ROOT/gtest/$(basename $FILE)
done
