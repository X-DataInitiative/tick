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
pushd $CWD/.. 2>&1 > /dev/null
ROOT=$PWD
popd 2>&1 > /dev/null

source $ROOT/sh/configure_env.sh

COMPILE=1
LINK=1
RELEASE=${RELEASE:-0}

CLI_ARGS=( "$@" )
CLI_ARGS_LEN=${#CLI_ARGS[@]}
if (( $CLI_ARGS_LEN > 0 )); then
  NPROFILES=()
  for (( i=1; i<${CLI_ARGS_LEN}+1; i++ )); do
    C=${CLI_ARGS[$i-1]}
    case $C in
        -c=*|--compile=*)
        COMPILE="${C#*=}"
        shift # past argument=value
        ;;
        -l=*|--link=*)
        LINK="${C#*=}"
        shift # past argument=value
        ;;
        *)
        NPROFILES+=("$C")
        ;;
    esac
  done
  NPROFILES_LEN=${#NPROFILES[@]}
  (( $NPROFILES_LEN > 0 )) && PROFILES=("${NPROFILES[@]}")
fi

TICK_INDICES=$($PY $ROOT/tools/python/numpy/check_sparse_indices.py)
MKN_C_FLAGS=" $TICK_INDICES"

[ -n "$CXXFLAGS" ] && MKN_C_FLAGS+=" ${CXXFLAGS} "
[ "$DEBUG" == "1" ] && MKN_C_FLAGS+=" -DDEBUG_COSTLY_THROW"

[ -n "$SWIG" ] && [ "$SWIG" != "0" ] && source $ROOT/sh/swig.sh

cd $ROOT

MKN_PROFILES=$(IFS=,; echo "${PROFILES[*]}")
(( $COMPILE == 1 )) && \
  MKN_BETA=1 $MKN compile -Dt ${MKN_COMPILE_THREADS} -a "${MKN_C_FLAGS[@]}" -b "$PY_INCS" \
      ${MKN_X_FILE[@]} -p ${MKN_PROFILES} -C lib "${MKN_WITH[@]}" -g $MKN_G -O $MKN_O

if (( $IS_WINDOWS )); then
  export MKN_LIB_EXT=".pyd"
else
  (( $RELEASE )) && export MKN_LIB_PRE=""
fi

(( $LINK == 1 )) && \
  for P in "${PROFILES[@]}"; do
      EX=$(hash_index $P)
      LIBLD=${LIB_LD_PER_LIB[$EX]}

      $MKN link -Sp $P -l "${LIBLDARGS} ${LDARGS} $LIBLD" \
         -P "${MKN_P}" -C lib "${MKN_WITH[@]}" \
         ${MKN_X_FILE[@]} -B "$B_PATH" -g $MKN_G -O $MKN_O

      if (( $RELEASE == 0 )); then
        PUSHD=${LIBRARIES[$EX]}
        pushd $ROOT/$(dirname ${PUSHD}) 2>&1 > /dev/null
          if (( $IS_WINDOWS == 0 )); then
            # regular lib* kept for gtest linking, no gtest during release
            for f in $(find . -maxdepth 1 -type f ! -name "*.py" ! -name "__*" ); do
              SUB=${f:2:3}
              [ "$SUB" == "lib" ] && cp "$f" "${f:5}"
            done
          fi
        popd 2>&1 > /dev/null
      fi
  done

echo "Finished - Success"
