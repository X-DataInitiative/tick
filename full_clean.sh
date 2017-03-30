#!/usr/bin/env bash

rm -r build/*

find . -name '*.so' -delete
find . -name '*_wrap.cpp' -delete
find . -name '*.pyc' -delete
