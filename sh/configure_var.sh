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

# We get the filename expected for C++ shared libraries
LIB_POSTFIX=$($PY -c "import distutils; from distutils import sysconfig; print(sysconfig.get_config_var('SO'))")

unameOut="$(uname -s)"

function relpath(){
  echo $($PY -c "import os.path; print(os.path.relpath('$1', '$2'))")
}
function linkread(){
if [[ "$unameOut" == "CYGWIN"* ]] || [[ "$unameOut" == "MINGW"* ]] || [[ "$unameOut" == "MSYS_NT"* ]]; then
  echo $(readlink -f $1)
else
  echo $($PY -c "import os; print(os.path.realpath(\"$1\"))")
fi
}

#################
if [[ "$unameOut" == "CYGWIN"* ]] || [[ "$unameOut" == "MINGW"* ]] || [[ "$unameOut" == "MSYS_NT"* ]]; then
  function pathreal(){
    P=$1
    which cygpath.exe 2>&1 > /dev/null
    CYGPATH=$?
    (( $CYGPATH == 0 )) && P=$(cygpath -w $P)
    echo "$P" | sed -e "s/\\\/\\\\\\\/g"
  }
else
  function pathreal(){
    echo $1
  }
fi
#################

# if windows - python-config does not exist
if [[ "$unameOut" == "CYGWIN"* ]] || [[ "$unameOut" == "MINGW"* ]] || [[ "$unameOut" == "MSYS_NT"* ]]; then
  echo "Windows detected"

  CL_PATH=0
  GCC_PATH=0
  which cl.exe &> /dev/null && CL_PATH=$(which cl.exe)
  which gcc.exe &> /dev/null && GCC_PATH=$(which gcc.exe)
  if [[ $CL_PATH == 0 ]] && [[ $GCC_PATH == 0 ]] ; then
    echo "Neither cl.exe or gcc.exe on path: Error"
    exit 1
  fi

  if [ -n "$CL_PATH" ]; then
    # Msys and cygwin can have their own "link.exe"
    # which interferes with MSVC link.exe
    PATH="$(dirname $CL_PATH):$PATH"
  fi

  PYDIR=$(dirname $(which $PY))
  PYINC="$PYDIR/include"
  [ ! -d "$PYINC" ] && \
      echo "$PYNUMINC does not exist - python in inconsistent state - reinstall"

  PYNUMINC="$PYDIR/Lib/site-packages/numpy/core/include"
  [ ! -d "$PYNUMINC" ] &&  \
      echo "$PYNUMINC does not exist - install numpy"

  # Deducing include paths for python and numpy
  [ -z "$PINC" ] && PINC="$PYINC"
  [ -z "$PNIC" ] && PNIC="$PYNUMINC";

  PINC_DRIVE=$(echo "$PINC" | cut -d'/' -f2)
  PINC="${PINC_DRIVE}:/${PINC:3}"
  PNIC="${PINC_DRIVE}:/${PNIC:3}"

  PY_INCS="${PINC};${PNIC}"

  B_PATH="$PYDIR/libs"
  [ -z "$LIB_POSTEXT" ] && LIB_POSTEXT="lib"

else
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

  PY_INCS="${PINC}:${PNIC}"

  LDARGS="$($PCONF --ldflags)"
  B_PATH="."
  [ -z "$LIB_POSTEXT" ] && LIB_POSTEXT="${LIB_POSTFIX##*.}"
fi
LIB_POSTFIX="${LIB_POSTFIX%.*}"

# Finding the installed version of "swig" to later find
# To override export SWIG variable with full path to swig binary
[ -z "$SWIG"  ] && which swig &> /dev/null && SWIG="swig"
[ -z "$SWIG"  ] && echo "swig not found on PATH, exiting with error" && exit 1;
[ -z "$SWIG_C_FLAGS" ] && SWIG_C_FLAGS="-DDEBUG_COSTLY_THROW"

# Here: "sed" is used to remove unnecessary parts of the link flags
#  given to us by python-config
LDARGS=$(echo "$LDARGS" | sed -e "s/ -lintl//g")
LDARGS=$(echo "$LDARGS" | sed -e "s/ -ldl//g")
LDARGS=$(echo "$LDARGS" | sed -e "s/ -framework//g")
LDARGS=$(echo "$LDARGS" | sed -e "s/ -Wl,-stack_size,1000000//g")
LDARGS=$(echo "$LDARGS" | sed -e "s/ CoreFoundation//g")

[ ! -z "$LDFLAGS" ] && LDARGS="${LDARGS} ${LDFLAGS}"

# IFS = Internal Field Separator - allows looping over lines with spaces
IFS=$'\n'

ANACONDA=0
LIBLDARGS=""
# Add library path if using anaconda python
PYVER=$($PY --version 2>&1 )
if [[ "$PYVER" == *"Anaconda"* ]]; then
  ANACONDA=1
  ANACONDA_LIB_PATH=$(linkread $(dirname $(which $PY))/../lib)
  if [[ "$unameOut" == "Darwin"* ]]; then
    ## Unknown reason why these are not required for Anaconda -
    ##  Not just that but neither is the -lpython
    ##  The line below is left intentionally commented as a reference of what should be linked
    LDARGS="" #-L${ANACONDA_LIB_PATH} -Wl,-rpath,${ANACONDA_LIB_PATH} $LDARGS -lmkl_rt -lpthread"
    LIBLDARGS="-dynamiclib -undefined dynamic_lookup -Wl,-headerpad_max_install_names"
  fi
fi

PROFILES=(
    array
    base
    base_model
    random
    linear_model
    prox
    solver
    hawkes/simulation
    hawkes/model
    hawkes/inference
    preprocessing
    robust
    survival
    array_test
)
function hash_index() {
    case $1 in
        'array')              echo 0;;
        'base')               echo 1;;
        'base_model')         echo 2;;
        'random')             echo 3;;
        'linear_model')       echo 4;;
        'prox')               echo 5;;
        'solver')             echo 6;;
        'hawkes/simulation')  echo 7;;
        'hawkes/model')       echo 8;;
        'hawkes/inference')   echo 9;;
        'preprocessing')      echo 10;;
        'robust')             echo 11;;
        'survival')           echo 12;;
        'array_test')         echo 13;;
    esac
}
LIBRARIES=(
    "tick/array/build/_array$LIB_POSTFIX"
    "tick/base/build/_base$LIB_POSTFIX"
    "tick/base_model/build/_base_model$LIB_POSTFIX"
    "tick/random/build/_crandom$LIB_POSTFIX"
    "tick/linear_model/build/_linear_model$LIB_POSTFIX"
    "tick/prox/build/_prox$LIB_POSTFIX"
    "tick/solver/build/_solver$LIB_POSTFIX"
    "tick/hawkes/simulation/build/_hawkes_simulation$LIB_POSTFIX"
    "tick/hawkes/model/build/_hawkes_model$LIB_POSTFIX"
    "tick/hawkes/inference/build/_hawkes_inference$LIB_POSTFIX"
    "tick/preprocessing/build/_preprocessing$LIB_POSTFIX"
    "tick/robust/build/_robust$LIB_POSTFIX"
    "tick/survival/build/_survival$LIB_POSTFIX"
    "tick/array_test/build/array_test${LIB_POSTFIX}"
)
