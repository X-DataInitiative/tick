#!/usr/bin/env bash

set -e

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PY=python
VER=$($PY --version 2>&1 | cut -d' ' -f 2 | cut -d'.' -f1)
((VER == 2 )) && PY=python3

cd ..
$PY -m yapf --style tools/python/yapf.conf -i tick --recursive

echo "Please confirm your version of yapf is up to date from pip"
echo "To upgrade run 'pip3 install yapf --upgrade'"
