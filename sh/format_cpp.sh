#!/usr/bin/env bash

set -e

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT=$CWD/..
cd $ROOT

DIRS=(lib/include lib/cpp lib/cpp-test)

[ -z "$CLANG_FORMAT" ] && which clang-format &> /dev/null \
                       && CLANG_FORMAT="clang-format"
[ -z "$CLANG_FORMAT" ] && echo "clang-format not on PATH, " \
                               "add to PATH or set CLANG_FORMAT" \
                                && exit 1

[ -z "$CLANG_FORMATDIFF" ] && which clang-format-diff.py &> /dev/null \
                           && CLANG_FORMATDIFF="clang-format-diff.py"
[ -z "$CLANG_FORMATDIFF" ] && echo "clang-format-diff.py not on PATH, " \
                                   "add to PATH or set CLANG_FORMATDIFF" \
                                    && exit 1

BRANCH=$(git symbolic-ref HEAD | sed -e 's,.*/\(.*\),\1,')

echo "Using $CLANG_FORMAT and $CLANG_FORMATDIFF on branch $BRANCH"

for dir in ${DIRS[@]}; do
  git diff $dir | python2 $CLANG_FORMATDIFF -binary $CLANG_FORMAT -p1 -i -style=file
done

# # Uncomment to run on all includes/sources
# pushd $CWD/..
# for f in $(find lib/cpp* -type f \( -iname \*.h -o -iname \*.cpp \)); do
#   $CLANG_FORMAT  -i -style=file $f
# done
# for f in $(find lib/include -type f \( -iname \*.h -o -iname \*.cpp \)); do
#   $CLANG_FORMAT  -i -style=file $f
# done
# popd
# # Uncomment to run on all includes/sources
