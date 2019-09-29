#!/usr/bin/env bash

set -e -x

python_versions=(cp36-cp36m cp37-cp37m)

for PYVER in ${python_versions[@]}; do
    PYBIN=/opt/python/${PYVER}/bin

    cd /io
    "${PYBIN}/python3" -m pip install pip --upgrade
    "${PYBIN}/python3" -m pip install -r requirements.txt
    "${PYBIN}/python3" setup.py bdist_wheel --dist-dir=/tick/wheelhouse
done

for whl in /tick/wheelhouse/*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done