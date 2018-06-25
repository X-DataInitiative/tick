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

python setup.py cpplint build_ext --inplace cpptest pytest
export PYTHONPATH=${PYTHONPATH}:`pwd`
for f in $(find examples -maxdepth 1 -type f -name "*.py"); do
  FILE=$(basename $f)
  FILE="${FILE%.*}"
  DISPLAY="-1" python -c "import tick.base; import examples.$FILE"
done
