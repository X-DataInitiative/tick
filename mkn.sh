#!/usr/bin/env bash
#
# Author: Philip Deegan
# Email : philip.deegan@polytechnique.edu 
# Date  : 21 - September - 2017
#
# This script compiles tick C++ libs in parallel
#   with optimally detected number of threads
#
# Requires the mkn build tool
#
# Input arguments are optional but if used are to be the 
#  modules to be compile, otherwise all modules are
#  compiled - use command "mkn profiles" to view available 
#  modules/profiles
#
# ccache is recommended
#
######################################################################

set -e

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT=$PWD

source $ROOT/sh/configure_env.sh

CLI_ARGS=( "$@" )
CLI_ARGS_LEN=${#CLI_ARGS[@]}
if (( $CLI_ARGS_LEN > 0 )); then
    PROFILES=()
    ALL=0
    for (( i=1; i<${CLI_ARGS_LEN}+1; i++ ));
    do
       PROFILES+=("${CLI_ARGS[$i-1]}")
    done
fi

TICK_INDICES=$($PY $ROOT/tools/python/numpy/check_sparse_indices.py)
MKN_C_FLAGS=" $TICK_INDICES "

[ -n "$CXXFLAGS" ] && MKN_C_FLAGS+=" ${CXXFLAGS} "
[ "$DEBUG" == "1" ] && MKN_C_FLAGS+=" -DDEBUG_COSTLY_THROW"

source $ROOT/sh/swig.sh

for P in "${PROFILES[@]}"; do 
    mkn compile -ta "${MKN_C_FLAGS[@]}" -b "$PY_INCS" -p ${P} 
done

TKLOG=$KLOG
for P in "${PROFILES[@]}"; do
    EX=$(hash_index $P)
    LIBLD=${LIB_LD_PER_LIB[$EX]}
    if [[ "$unameOut" == "CYGWIN"* ]] || [[ "$unameOut" == "MINGW"* ]]; then
      # Here we intercept the command for linking on windows and change 
      # the output from ".dll" to ".pyd"
      KLOG=0 
      OUT=$(mkn link -p $P -l "${LDARGS} $LIBLD" \
            -P lib_name=$LIB_POSTFIX \
            -RB $B_PATH |  head -1)
      OUT=$(echo $OUT | sed -e "s/.dll/.pyd/g")
      (( TKLOG > 0 )) && echo $OUT
      cmd /c "${OUT[@]}"
    else
      mkn link -p $P -l "${LDARGS} $LIBLD" \
         -P lib_name=$LIB_POSTFIX \
         -B $B_PATH
    fi
    PUSHD=${LIBRARIES[$EX]}
    pushd $(dirname ${PUSHD}) 2>&1 > /dev/null
      if [[ "$unameOut" == "CYGWIN"* ]] || [[ "$unameOut" == "MINGW"* ]]; then
        for f in $(find . -maxdepth 1 -type f -name "*.dll" ); do    
          DLL="${f%.*}"
          cp ${f:2} ${DLL}.pyd
        done
      else
        for f in $(find . -maxdepth 1 -type f ! -name "*.py" ! -name "__*" ); do
          SUB=${f:2:3}
          [ "$SUB" == "lib" ] && cp "$f" "${f:5}"
        done
      fi      
    popd 2>&1 > /dev/null
done
