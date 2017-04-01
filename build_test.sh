#!/usr/bin/env bash

## Using the python executable defined in env. variable TICK_PYTHON otherwise fall back to "python"
PYTHON_EXEC=${TICK_PYTHON:=python}

${PYTHON_EXEC} setup.py build_ext --inplace pytest

