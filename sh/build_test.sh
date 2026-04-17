#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

## Using the python executable defined in env. variable TICK_PYTHON otherwise fall back to "python"
PYTHON_EXEC=${TICK_PYTHON:=python}

cd "$CWD/.."
${PYTHON_EXEC} -m pip install -e ".[dev]"
${PYTHON_EXEC} -m pytest -q
