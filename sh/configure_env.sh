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

echo "Entering configure_env.sh"

LIB_LD_PER_LIB=()
ITER=0
for PROFILE in "${PROFILES[@]}"; do
  LIBS=
  TREE=($(mkn tree -p $PROFILE -C lib))
  TREE_LEN=${#TREE[@]}
  for idx in $(seq $TREE_LEN -1 0); do
    LINE="${TREE[idx]}"
    set +e
    echo "$LINE" | grep "+" | grep "tick" 2>&1 > /dev/null
    WIN=$?
    set -e
    if [[ "$WIN" == "0" ]]; then
      INDEX=$(echo "$LINE" | cut -d '[' -f2 | cut -d ']' -f1)
      EX=$(hash_index $INDEX)
      ADD_LIB=${LIBRARIES[$EX]}
      if [ -n "$ADD_LIB" ]; then
        if [ -n "$LIBS" ]; then
          set +e
          echo "$LIBS" | grep $(basename "$ADD_LIB") 2>&1 > /dev/null
          WIN=$?
          set -e
          [[ "$WIN" == "0" ]] && continue
        fi

        if [[ "$unameOut" == "Darwin"* ]]; then
          LIBS="$LIBS $(linkread ${ADD_LIB}.${LIB_POSTEXT})"
          REL=$(dirname ${LIBRARIES[$ITER]})
          RPATH=$(dirname $ADD_LIB)
          if [ "$REL" != "$PATH" ]; then
            LIBS="$LIBS -Wl,-rpath,@loader_path/$(relpath $RPATH $REL)"
          fi
        else
          LIBS="$LIBS $(pathreal $(linkread ${ADD_LIB}.${LIB_POSTEXT}))"
        fi
      fi
    fi
  done
  if [[ "$unameOut" == "Darwin"* ]]; then
    FIL=$(basename ${LIBRARIES[$ITER]})
    CLANG_LARGS+=" -Wl,-install_name,@rpath/${FIL}.${LIB_POSTEXT}"
  fi
  LIB_LD_PER_LIB+=("$LIBS")
  ITER=$(($ITER + 1))
done


##
# The mkn -x option overrides the default settings file
#  to allow to build with differernt compilers/configurations
MKN_X="${MKN_X_FILE}"
[ -n "${MKN_X_FILE}" ] && MKN_X_FILE=(-x $MKN_X)

##
# We allow for compiler specific linking arguments
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd $THIS_DIR/.. 2>&1 > /dev/null
COMMAND=$(mkn compile -Rp array ${MKN_X_FILE[@]} -C lib | head -1 )
if [[ "$COMMAND" == *"icpc "* ]]; then
  LDARGS+="$(echo " -Wl,-rpath,${ROOT}")"
fi
popd 2>&1 > /dev/null
#
##

##
# MKN_P_ARRAY exists to allow platfrom
#  specific properties to be passed to mkn if required
MKN_P_ARRAY=(lib_name=$LIB_POSTFIX)
MKN_P=""
for PROP in "${MKN_P_ARRAY[@]}"; do
  MKN_P+=${PROP},
done
MKN_P_SIZE=${#MKN_P}
MKN_P_SIZE=$((MKN_P_SIZE - 1))
X_MKN_P="${MKN_P:0:${MKN_P_SIZE}}"
MKN_P=($X_MKN_P)

# The argument passed to "mkn -P" is "MKN_P"
#  such that all entries in the MKN_P_ARRAY
#  become CSV values in MKN_P
#  Any commas (,) in array entries must
#  be escaped with a single backslash (\)
##

V_MKN_WITH="$MKN_WITH"
MKN_WITH_SIZE=${#V_MKN_WITH}
(( $MKN_WITH_SIZE > 0 )) && MKN_WITH=(-w $V_MKN_WITH)

##
# if a file included from source ends with a non-zero exitting command
#  the "set -e" can cause the script to exit


export TICK_CONFIGURED=1

echo "Finished configure_env.sh"
##


