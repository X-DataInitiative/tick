#!/usr/bin/env bash
# Author: Philip Deegan / philip.deegan@gmail.com
set -ex
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd $CWD/../lib
(cd bin && cp -r tick.py gtest_lib)
mkn link -p gtest_lib
mkn build -p gtest -d google.test -Ota "$@"
cd "$CWD"
chmod +x gtest.sh
./gtest.sh
