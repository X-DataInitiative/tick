FROM quay.io/pypa/manylinux1_x86_64
WORKDIR /tick

RUN yum update -y && yum install -y zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel openssl-devel pcre-devel atlas-devel

ENV PATH="/root/.pyenv/bin:$PATH" SWIG_VER=3.0.12

# Installing pyenv
RUN curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash

# Installing swig
RUN curl -O https://kent.dl.sourceforge.net/project/swig/swig/swig-${SWIG_VER}/swig-${SWIG_VER}.tar.gz && tar -xf swig-${SWIG_VER}.tar.gz && \
	cd swig-${SWIG_VER} && ./configure && make && make install && \
	rm -rf swig-${SWIG_VER}.tar.gz swig-${SWIG_VER} 

# Installing cmake
RUN curl -O https://cmake.org/files/v3.8/cmake-3.8.0.tar.gz && tar -xf cmake-3.8.0.tar.gz && \
    (cd cmake-3.8.0 && ./configure && gmake && gmake install) && \
    rm -rf cmake-3.8.0.tar.gz cmake-3.8.0

# Installing googletest
RUN git clone https://github.com/google/googletest.git && \
	(cd googletest && mkdir -p build && cd build && cmake .. && make && make install) && \
	rm -rf googletest

# Installing necessary python versions
RUN pyenv install 3.4.6 && pyenv install 3.5.3 && pyenv install 3.6.1

COPY requirements.txt requirements.txt

RUN eval "$(pyenv init -)" && \
	pyenv local 3.4.6 && pip install -r requirements.txt -U pip && \
	pyenv local 3.5.3 && pip install -r requirements.txt -U pip && \
	pyenv local 3.6.1 && pip install -r requirements.txt -U pip

LABEL maintainer "soren.poulsen@polytechnique.edu"