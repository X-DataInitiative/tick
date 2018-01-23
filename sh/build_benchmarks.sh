#!/bin/bash

set -e

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $CWD/.. 
ROOT=$PWD

mkdir -p $ROOT/build/bench && cd $ROOT/build/bench

rm -rf build_noopt && mkdir -p build_noopt

CMAKE_DIR="${ROOT}/lib"
echo ${CMAKE_DIR}

builds=(build_noopt )

printf "\nNo optimization build\n"
(cd build_noopt && \
  cmake VERBOSE=1 -DCMAKE_BUILD_TYPE=Release ${CMAKE_DIR})

if [ -z "$SKIP_BLAS" ]; then
  printf "\nBLAS build\n"
  rm -rf build_blas && mkdir -p build_blas
  (cd build_blas && \
    cmake VERBOSE=1 -DUSE_BLAS=ON -DCMAKE_BUILD_TYPE=Release ${CMAKE_DIR})
  builds+=(build_blas)
fi

if [ -z "$SKIP_MKL" ]; then
  printf "\nMKL build\n"
  rm -rf build_mkl && mkdir -p build_mkl
  (cd build_mkl && \
    cmake VERBOSE=1 -DUSE_MKL=ON -DCMAKE_BUILD_TYPE=Release ${CMAKE_DIR})
  builds+=(build_mkl)
fi

mkdir -p $ROOT/build/bench && cd $ROOT/build/bench

# Use only one core if no argument is provided
N_CORES=${1:-1}
for d in "${builds[@]}" ; do echo ${d} && (cd ${d} && make -j${N_CORES}); done
