#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "CLEAN"
$CWD/full_clean.sh

$CWD/build_test.sh
