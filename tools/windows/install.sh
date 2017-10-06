#! /bin/env bash

# install msys2 to use this @ http://www.msys2.org

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TOP=$(readlink -f $CWD/../..)
DWNLDD=$TOP/downloaded

P_MAJ="3"
P_MIN="6"
P_VER="$P_MAJ.$P_MIN.2"
NUMKL_VER="1.13.*"
NUMDC_VER="0.7.0"
SCIPY_VER="0.19.*"

PPATH=0
SPATH=0

CURL=0

findWHL(){
    RET=$(find $1 -name $2 | wc -l)
    echo $RET
}

which curl.exe &> /dev/null && CURL=$(which curl.exe)
[ $CURL == 0 ] && echo "curl not found, cannot download installers"

source $CWD/check_python.sh
source $CWD/check_swig.sh

PIP=$(dirname $PPATH)/Scripts/pip.exe
[ ! -f $PIP ] && echo "ERROR: PIP not found @ $PIP" && exit 1

SO=$($PPATH -c "import distutils; from distutils import sysconfig; print(sysconfig.get_config_var('SO'))")
WHEEL="${SO:6:-4}"

NUMLK="numpy-$NUMKL_VER+mkl-cp${P_MAJ}${P_MIN}-cp${P_MAJ}${P_MIN}m-${WHEEL}.whl"
NUMDC="numpydoc-$NUMDC_VER-py2.py3-none-any.whl"
SCIPY="scipy-$SCIPY_VER-cp${P_MAJ}${P_MIN}-cp${P_MAJ}${P_MIN}m-${WHEEL}.whl"

if [[ $(findWHL $DWNLDD "$NUMLK") -eq 0 ]] ; then
    echo "please download $NUMLK to $DWNLDD and rerun this script"
    echo "From: http://www.lfd.uci.edu/~gohlke/pythonlibs"
    exit 1
fi
NUMLK=$(basename $(find $DWNLDD -name "$NUMLK"))

if [[ $(findWHL $DWNLDD "$SCIPY") -eq 0 ]] ; then
    echo "please download $SCIPY to $DWNLDD and rerun this script"
    echo "From: http://www.lfd.uci.edu/~gohlke/pythonlibs"
    exit 1
fi
SCIPY=$(basename $(find $DWNLDD -name "$SCIPY"))

if [[ $(findWHL $DWNLDD "$NUMDC") -eq 0 ]] ; then
    echo "please download $NUMDC to $DWNLDD and rerun this script"
    echo "From: http://www.lfd.uci.edu/~gohlke/pythonlibs"
    exit 1
fi
NUMDC=$(basename $(find $DWNLDD -name "$NUMDC"))

$PIP install wheel
$PIP install $DWNLDD/$NUMLK
$PIP install $DWNLDD/$NUMDC
$PIP install $DWNLDD/$SCIPY

$PIP install -r $TOP/requirements.txt
$PPATH -m pip install --upgrade setuptools

echo ""
echo "Attempting to compile: vcvars.bat must be called manually prior to this"
echo ""

PATH=$(dirname $PPATH):$(dirname $SPATH):$PATH python setup.py build
