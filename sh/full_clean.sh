#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

## Using the python executable defined in env. variable TICK_PYTHON otherwise fall back to "python"
PYTHON_EXEC=${TICK_PYTHON:=python3}

rm -rf $CWD/../build
rm -rf $CWD/../dist
rm -rf $CWD/../lib/bin
find $CWD/.. -maxdepth 2 -name '*.egg-info' -exec rm -rf {} +
