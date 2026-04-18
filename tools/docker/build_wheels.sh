#!/usr/bin/env bash

set -e -x

python_versions=(cp311-cp311 cp312-cp312 cp313-cp313)

for PYVER in ${python_versions[@]}; do
    PYBIN=/opt/python/${PYVER}/bin

    cd /io
    "${PYBIN}/python3" -m pip install --upgrade pip
    "${PYBIN}/python3" -m pip install build
    "${PYBIN}/python3" -m build --wheel --outdir /tick/wheelhouse
done

for whl in /tick/wheelhouse/*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done
