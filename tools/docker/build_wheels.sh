#!/usr/bin/env bash

set -e -x

python_versions=(cp34-cp34m  cp35-cp35m  cp36-cp36m)

for PYVER in ${python_versions[@]}; do
    PYBIN=/opt/python/${PYVER}/bin

    cd /io

    "${PYBIN}/python3" -mpip install -r requirements.txt
    "${PYBIN}/python3" setup.py bdist_wheel --dist-dir=/tick/wheelhouse
done

for whl in /tick/wheelhouse/*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done