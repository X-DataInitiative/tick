#
# Author: Philip Deegan
# Email : philip.deegan@polytechnique.edu 
# Date  : 21 - September - 2017
#
# This script is to be included to setup variables to build tick
#   It is done so to be cross platform and heterogeneous
#
# Variables
#  $PY = "python" binary
#  $PCONF = "python-config" binary
#  $SWIG  = "swig" binary
#  $LDARGS=Linking arguments
#  $LIB_POSTFIX= python library format 
#  $LIB_POSTEXT= system library file extention
#  $PROFILES= array of mkn profiles to build
# 
######################################################################

set -e

TICK_CONFIGURED=1

# Finding the system python binary
# To override, export PY variable with full path to python binary
[ -z "$PY" ] && which python3 &> /dev/null && PY="python3"
[ -z "$PY" ] && which python &> /dev/null && PY="python"
[ -z "$PY" ] && echo "python or python3 not on path, exiting with error" && exit 1

PYVER=$($PY -c "import sys; print(sys.version_info[0])")
if (( $PYVER < 3)); then
    echo "Python version 3 is required, if installed export PY variable - e.g. export PY=/path/to/python3"
    echo "ERROR: 1"
    exit 1
fi

PYVER_MIN=$($PY -c "import sys; print(sys.version_info[1])")
        
# Finding the installed version of "python-config" to later find 
#  include paths and linker flags
# To override export PCONF variable with full path to python-config binary
[ -z "$PCONF"  ] && which python3-config &> /dev/null && PCONF="python3-config"
[ -z "$PCONF"  ] && which python3.${PYVER_MIN}-config &> /dev/null && PCONF="python3.${PYVER_MIN}-config"
[ -z "$PCONF"  ] && which python-config &> /dev/null && PCONF="python-config" \
       && echo "WARNING: python-config may be the incorrect version!"
[ -z "$PCONF"  ] && echo "python-config or python3-config not found on PATH, exiting with error" && exit 1;

# Deducing include paths for python and numpy
[ -z "$PINC" ] && for i in $($PCONF --includes); do PINC="${PINC}${i:2}:"; done
[ -z "$PNIC" ] && PNIC=$($PY -c "import numpy as np; print(np.get_include())");

# Finding the installed version of "swig" to later find 
# To override export SWIG variable with full path to swig binary
[ -z "$SWIG"  ] && which swig &> /dev/null && SWIG="swig"
[ -z "$SWIG"  ] && echo "swig not found on PATH, exiting with error" && exit 1;

[ -z "$SWIG_C_FLAGS" ] && SWIG_C_FLAGS="-DDEBUG_COSTLY_THROW"


# Here: "sed" is used to remove unnecessary parts of the link flags 
#  given to us by python-config
LDARGS="$($PCONF --ldflags)"
LDARGS=$(echo "$LDARGS" | sed -e "s/ -lintl//g")
LDARGS=$(echo "$LDARGS" | sed -e "s/ -ldl//g")
LDARGS=$(echo "$LDARGS" | sed -e "s/ -framework//g")
LDARGS=$(echo "$LDARGS" | sed -e "s/ CoreFoundation//g")

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     LDARGS="${LDARGS} $(mkn -G nix_largs)";;
    Darwin*)    LDARGS="${LDARGS} $(mkn -G bsd_largs)";;
    CYGWIN*)    LDARGS="${LDARGS}";;
    MINGW*)     LDARGS="${LDARGS}";;
    *)          LDARGS="${LDARGS}";;
esac
[ ! -z "$LDFLAGS" ] && LDARGS="${LDARGS} ${LDFLAGS}"

# IFS = Internal Field Separator - allows looping over lines with spaces
IFS=$'\n'

LIB_POSTFIX=$($PY -c "import distutils; from distutils import sysconfig; print(sysconfig.get_config_var('SO'))")
LIB_POSTEXT="${LIB_POSTFIX##*.}"
LIB_POSTFIX="${LIB_POSTFIX%.*}"

function linkread(){
    echo $($PY -c "import os; print(os.path.realpath(\"$1\"))")
}
function pathreal(){
    if [[ $1 == *"$CWD"* ]]; then
        IN=$1
        LEN=${#CWD}
        echo \.${IN:$LEN}
    else
        echo $1
    fi
}

PROFILES=(
    array
    base
    random
    optim/model
    optim/prox
    optim/solver
    simulation
    inference
    preprocessing
    array_test
)
function hash_index() {
    case $1 in
        'array')         echo 0;;
        'base')          echo 1;;
        'random')        echo 2;;
        'optim/model')   echo 3;;
        'optim/prox')    echo 4;;
        'optim/solver')  echo 5;;
        'simulation')    echo 6;;
        'inference')     echo 7;;
        'preprocessing') echo 8;;
        'array_test')    echo 9;;
    esac
}
LIBRARIES=(
    "tick/base/array/build/_array$LIB_POSTFIX"
    "tick/base/build/_base$LIB_POSTFIX"
    "tick/random/build/_crandom$LIB_POSTFIX"
    "tick/optim/model/build/_model$LIB_POSTFIX"
    "tick/optim/prox/build/_prox$LIB_POSTFIX"
    "tick/optim/solver/build/_solver$LIB_POSTFIX"
    "tick/simulation/build/_simulation$LIB_POSTFIX"
    "tick/inference/build/_inference$LIB_POSTFIX"
    "tick/preprocessing/build/_preprocessing$LIB_POSTFIX"
    "tick/base/array_test/build/array_test${LIB_POSTFIX}"
)

