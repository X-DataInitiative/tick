#!/usr/bin/env bash

set -e -x

shell_session_update() { :; }

eval "$(pyenv init -)"

pyenv global ${PYVER}

### Commented to reduce time on travis
## python -m pip install yapf --upgrade
## python -m yapf --style tools/code_style/yapf.conf -i tick --recursive
## (( $(git diff tick | wc -l) > 0 )) && \
## echo "Python has not been formatted : Please run ./sh/format_python.sh and recommit" \
##   && exit 2
## export PYTHONPATH=${PYTHONPATH}:`pwd` && cd doc && make doctest
###

python setup.py cpplint
set +e
python setup.py build_ext -j 2 --inplace # mac builds are slow but -j can fail so run twice
rm -rf build/lib # force relinking of libraries in case of failure
set -e
python setup.py build_ext --inplace cpptest pytest

export PYTHONPATH=${PYTHONPATH}:`pwd`
for f in $(find examples -maxdepth 1 -type f -name "*.py"); do
  FILE=$(basename $f)
  FILE="${FILE%.*}"
  DISPLAY="-1" python -c "import tick.base; import examples.$FILE"
done
