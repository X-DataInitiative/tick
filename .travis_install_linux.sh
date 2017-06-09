#!/usr/bin/env bash

sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test;
sudo apt-get -qq update
sudo apt-get -yqq install libblas-dev gfortran gcc-4.8 g++-4.8

export CC="gcc-4.8"
export CXX="g++-4.8"

if [ ! -d $HOME/.pyenv/bin ]
    then
        git clone https://github.com/yyuu/pyenv.git $HOME/.pyenv
fi

if [ ! -d googletest  ] || [ ! -f googletest/CMakeLists.txt ]
    then
        git clone https://github.com/google/googletest.git
        (cd googletest && mkdir -p build && cd build && cmake .. && make -s)
fi

if [ ! -d swig-3.0.10 ] || [ ! -f swig-3.0.10/configure ]
    then
        wget http://prdownloads.sourceforge.net/swig/swig-3.0.10.tar.gz && tar -xf swig-3.0.10.tar.gz
        (cd swig-3.0.10 && ./configure && make -s)
fi

(cd googletest && cd build && sudo make -s install)
(cd swig-3.0.10 && sudo make -s install)

alias swig=/usr/local/bin/swig

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"

eval "$(pyenv init -)"

pyenv install -s ${PYVER}