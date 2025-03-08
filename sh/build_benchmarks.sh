#!/bin/bash

set -ex

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $CWD/..
ROOT=$PWD
N_CORES=${N_CORES:=1}
CMAKE_DIR="${ROOT}/lib"

mkdir -p $ROOT/build/bench && cd $ROOT/build/bench
rm -rf build_noopt && mkdir -p build_noopt

echo ${CMAKE_DIR}

builds=( build_noopt )

printf "\nNo optimization build\n"
(cd build_noopt && \
  cmake VERBOSE=1 -DBENCHMARK=ON -DCMAKE_BUILD_TYPE=Release ${CMAKE_DIR})

if [ -z "$SKIP_BLAS" ]; then
  printf "\nBLAS build\n"
  rm -rf build_blas && mkdir -p build_blas
  (cd build_blas && \
    cmake VERBOSE=1 -DBENCHMARK=ON -DUSE_BLAS=ON -DCMAKE_BUILD_TYPE=Release ${CMAKE_DIR})
  builds+=(build_blas)
fi

if [ -z "$SKIP_MKL" ]; then
  printf "\nMKL build\n"
  rm -rf build_mkl && mkdir -p build_mkl
  (cd build_mkl && \
    cmake VERBOSE=1 -DBENCHMARK=ON -DUSE_MKL=ON -DCMAKE_BUILD_TYPE=Release ${CMAKE_DIR})
  builds+=(build_mkl)
fi

mkdir -p $ROOT/build/bench && cd $ROOT/build/bench

# Use only one core if no argument is provided

for d in "${builds[@]}" ; do (cd ${d} && make V=1 -j${N_CORES} VERBOSE=1); done
