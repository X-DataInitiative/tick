#!/usr/bin/env bash

set -ex

shell_session_update() { :; }

eval "$(pyenv init -)"

pyenv global ${PYVER}

python setup.py cpplint
python setup.py build_ext --inplace cpptest pytest

export PYTHONPATH=${PYTHONPATH}:`pwd`
for f in $(find examples -maxdepth 1 -type f -name "*.py"); do
  FILE=$(basename $f)
  FILE="${FILE%.*}"    # skipping it takes too long
  [[ "plot_asynchronous_stochastic_solver" != "$FILE" ]] && \
    DISPLAY="-1" python -c "import tick.base; import examples.$FILE"
done
