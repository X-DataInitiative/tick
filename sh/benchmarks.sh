#!/usr/bin/env bash
#
# This script builds and launches benchmarks for tick
#
# Requires the mkn build tool
#
######################################################################

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $CWD/..
ROOT=$PWD
source $ROOT/sh/configure_env.sh

cd $CWD/..
export PYTHONPATH=$PWD

mkn clean build -p benchmarks -a "${CARGS}" \
    -tl "${LDARGS}" -b "$PINC:$PNIC" \
    -P lib_name=$LIB_POSTFIX \
    run

