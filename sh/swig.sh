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

echo "SWIG sub-routine - START"
# This block iterates over each module to create swig cpp files
for P in "${PROFILES[@]}"; do
    EX=$(hash_index $P)
    VAL="${LIBRARIES[$EX]}"
    DIR=$(pathreal "$(linkread $(dirname $VAL)/..)") 
    INCS=()
    IFS=$'\n'
    for j in $(mkn inc -p $P); do  
        if [[ $j != *"/third_party/"* ]]; then
            INCS+=(-I$(pathreal $j))
            SWIG_PATH="$(pathreal $(linkread $j)/../swig)"
            [ -d "$SWIG_PATH" ] && INCS+=(-I$SWIG_PATH)
        fi
    done
    if [[ $P == *"/"* ]]; then
        P=$(echo $P | cut -d'/' -f2)
    fi

    # Here we attempt to find files to "SWIG"
    #  Three different methods are attempted
    #   1. If there is only one file with ".i" extention, we process that file
    #   2. OR - All files ending in "module.i" are processed
    #   3. Lastly - if there is a ".i" file with the same name as the module
    #        it is processed
    IF=""
    if [ $(find $DIR/swig -name "*.i" | wc -l) == 1 ]; then
        IF=$(find $DIR/swig -name "*.i")
        B=$(basename $IF)
        B="${B%.*}"
        [ -n "$RESWIG" ] && (( "$RESWIG" == 1 )) && \
            [ -f "$DIR/swig/${B}_wrap.cpp" ] && \
            	rm "$DIR/swig/${B}_wrap.cpp"
        [ ! -f "$DIR/swig/${B}_wrap.cpp" ] && \
            $SWIG -python -py3 -c++ -modern -new_repr ${INCS[@]} \
            	-outdir $DIR/build "$SWIG_C_FLAGS" \
            	-o $DIR/swig/${B}_wrap.cpp $IF
    else
        for f in $(find $DIR/swig -name "*module.i"); do
            IF=$f;
            B=$(basename $IF) 
            B="${B%.*}"
            [ -n "$RESWIG" ] && (( "$RESWIG" == 1 )) && \
                [ -f "$DIR/swig/${B}_wrap.cpp" ] && \
            		rm "$DIR/swig/${B}_wrap.cpp"
            [ ! -f "$DIR/swig/${B}_wrap.cpp" ] && \
                $SWIG -python -py3 -c++ -modern -new_repr ${INCS[@]} \
                	-outdir $DIR/build "$SWIG_C_FLAGS" \
                	-o $DIR/swig/${B}_wrap.cpp $IF
        done
    fi
    if [ -z "$IF" ] && [ -f "${DIR}/swig/${P}.i" ]; then
        [ -n "$RESWIG" ] && (( "$RESWIG" == 1 )) && \
            [ -f "$DIR/swig/${P}_wrap.cpp" ] && \
        	    rm "$DIR/swig/${P}_wrap.cpp"    	
        [ ! -f "$DIR/swig/${P}_wrap.cpp" ] && \
        	$SWIG -python -py3 -c++ -modern -new_repr ${INCS[@]} \
        		-outdir $DIR/build "$SWIG_C_FLAGS" -o $DIR/swig/${P}_wrap.cpp \
        		"${DIR}/swig/${P}.i"
    fi
done
echo "SWIG sub-routine - END"
