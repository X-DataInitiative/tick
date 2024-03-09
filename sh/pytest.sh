#!/usr/bin/env bash
# Author: Philip Deegan / philip.deegan@gmail.com
set -ex
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd $CWD/..
CMD="""python3 -m unittest discover -v . "*_test.py""""
$CMD  || $CMD  || $CMD  # retry if failed
