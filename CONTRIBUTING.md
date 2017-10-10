

## Developers

We welcome developers and researchers to contribute to the _tick_ package. In order to do so, we ask that you submit pull requests that will be reviewed by the _tick_ team before being merged into the package source.

### Pull Requests

We ask that pull requests meet the following standards:

- PR contains exactly 1 commit that has clear and meaningful description
- All unit tests pass (Python and C++)
- C++ code follows our style guide ([Google style](https://google.github.io/styleguide/cppguide.html), this is tested if you use provided scripts, such as `./build_test.sh`)
- If new functionality is added, new unit-tests must be included in the pull request

In order to run these tests, you need:

- [Python unittest](https://docs.python.org/3/library/unittest.html)
- [gtest](https://github.com/google/googletest) 
- [cpplint](https://pypi.python.org/pypi/cpplint)

On most systems, this will suffice:

    # Install cpplint
    pip install cpplint

    # Install gtest
    git clone https://github.com/google/googletest.git
    (cd googletest && mkdir -p build && cd build && cmake .. && make && make install)

Our continuous integration tool (Travis CI) will also run these checks upon submission of a pull request.
