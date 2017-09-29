#!/usr/bin/env bash
#
# Author: Philip Deegan
# Email : philip.deegan@polytechnique.edu 
# Date  : 29 - September - 2017
#
# This script recreates swig cpp files for each module
# 
######################################################################

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $CWD/.. 
ROOT=$PWD

RESWIG=1

source $ROOT/sh/swig.sh
