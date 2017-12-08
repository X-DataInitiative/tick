#!/usr/bin/env bash
#
# Author: Philip Deegan
# Email : philip.deegan@polytechnique.edu 
# Date  : 29 - September - 2017
#
# This script creates swig cpp files for each module
# 
# Notes:
#   If the intended swig cpp file is found it is skipped
#   Unless, the environement variable "RESWIG" = 1
#    or call ./sh/reswig.sh
# 
######################################################################

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $CWD/.. 
ROOT=$PWD

if [ -z "$TICK_CONFIGURED" ]; then  
   source $ROOT/sh/configure_env.sh
fi

SWIG_BASE="lib/swig"

echo "SWIG sub-routine - START"
# This block iterates over each module to create swig cpp files
for P in "${PROFILES[@]}"; do

  LIBS=
  TREE=($(mkn tree -p $P -C lib))
  TREE_LEN=${#TREE[@]}
  INCS=(-Ilib/include)
  INCS+=(-I${SWIG_BASE}/base)
  INCS+=(-Ilib/swig/${P})
  for idx in $(seq $TREE_LEN -1 0); do
    LINE="${TREE[idx]}"
    set +e  
    echo "$LINE" | grep "+" | grep "tick" 2>&1 > /dev/null
    WIN=$?
    set -e
    if [[ "$WIN" == "0" ]]; then
      INDEX=$(echo "$LINE" | cut -d '[' -f2 | cut -d ']' -f1)
      INCS+=(-I${SWIG_BASE}/$INDEX)      
    fi
  done

  EX=$(hash_index $P)
  VAL="${LIBRARIES[$EX]}"
  DIR=$(pathreal "$(linkread $(dirname $VAL)/..)") 

  P1=$P
  if [[ $P == *"/"* ]]; then
      P1=$(echo $P | cut -d'/' -f2)
  fi

  # Here we attempt to find files to "SWIG"
  #  Three different methods are attempted
  #   1. If there is only one file with ".i" extention, we process that file
  #   2. OR - All files ending in "module.i" are processed
  #   3. Lastly - if there is a ".i" file with the same name as the module
  #        it is processed
  IF=""
  if [ $(find ${SWIG_BASE}/$P -name "*.i" | wc -l) == 1 ]; then
    IF=$(find ${SWIG_BASE}/$P -name "*.i")
    B=$(basename $IF)
    B="${B%.*}"
    [ -n "$RESWIG" ] && (( "$RESWIG" == 1 )) && \
        [ -f "${SWIG_BASE}/$P/${B}_wrap.cpp" ] && \
          rm "${SWIG_BASE}/$P/${B}_wrap.cpp"
    [ ! -f "${SWIG_BASE}/$P/${B}_wrap.cpp" ] && \
        $SWIG -python -py3 -c++ -modern -new_repr ${INCS[@]} \
          -outdir $DIR/build "$SWIG_C_FLAGS" \
          -o ${SWIG_BASE}/$P/${B}_wrap.cpp $IF
  else
    for f in $(find ${SWIG_BASE}/$P -name "*module.i"); do
        IF=$f;
        B=$(basename $IF) 
        B="${B%.*}"
        [ -n "$RESWIG" ] && (( "$RESWIG" == 1 )) && \
            [ -f "${SWIG_BASE}/$P/${B}_wrap.cpp" ] && \
            rm "${SWIG_BASE}/$P/${B}_wrap.cpp"
        [ ! -f "${SWIG_BASE}/$P/${B}_wrap.cpp" ] && \
            $SWIG -python -py3 -c++ -modern -new_repr ${INCS[@]} \
              -outdir $DIR/build "$SWIG_C_FLAGS" \
              -o ${SWIG_BASE}/$P/${B}_wrap.cpp $IF
    done
  fi
  if [ -z "$IF" ] && [ -f "${DIR}/swig/${P}.i" ]; then
    [ -n "$RESWIG" ] && (( "$RESWIG" == 1 )) && \
        [ -f "${SWIG_BASE}/$P/${P1}_wrap.cpp" ] && \
          rm "${SWIG_BASE}/$P/${P1}_wrap.cpp"     
    [ ! -f "${SWIG_BASE}/$P/${P}_wrap.cpp" ] && \
      $SWIG -python -py3 -c++ -modern -new_repr ${INCS[@]} \
        -outdir $DIR/build "$SWIG_C_FLAGS" -o ${SWIG_BASE}/$P/${P1}_wrap.cpp \
        "${DIR}/swig/${P1}.i"
  fi
done
echo "SWIG sub-routine - END"
