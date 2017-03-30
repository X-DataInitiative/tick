#!/usr/bin/env bash

## Using the python executable defined in env. variable TICK_PYTHON otherwise fall back to "python"
PYTHON_EXEC=${TICK_PYTHON:=python}

echo ""; echo "CHECK VERSIONS"
${PYTHON_EXEC} check_versions.py
if [ $? -eq 0 ]
    then
        echo "OK"
fi

echo ""; echo "BUILD"
exec > >(tee compilation_log.txt)
exec 2> >(tee error_log.txt)

## Directories to be checked by cpplint
CPPLINT_DIRS=(
    tick/base/array/src
    tick/base/array/tests/src
    tick/base/src
    tick/base/src/math
    tick/base/src/parallel
    tick/inference/src
    tick/optim/model/src
    tick/optim/solver/src
    tick/optim/prox/src
    tick/random/src
    tick/simulation/src
)

${PYTHON_EXEC} --version
${PYTHON_EXEC} setup.py build_ext --inplace

BUILD_RESULT=$?
BUILD_ERRORS=$(grep "error:" error_log.txt)

if [ ${BUILD_RESULT} -ne 0 ] || [ ! -z ${BUILD_ERRORS} ]
    then
        printf "\n\nBuild return code: ${BUILD_RESULT}. Error during compilation: \n${BUILD_ERRORS}\n";

        exit ${BUILD_RESULT}
    else exec > >(tee stdcouttest_log.txt); exec 2> >(tee test_log.txt)
        echo ""
        echo "C++ TEST"

        TICK_CPP_TEST_SUCCESS=0

        if type cmake > /dev/null
          then
            mkdir -p cpp_test_build

            (cd cpp_test_build && cmake -DCMAKE_BUILD_TYPE=Release ../tick && make && make check)
            TICK_CPP_TEST_RESULT=$?

            echo "C++ TEST DONE"
          else
            echo "CMake executable not found - not running C++ tests"
        fi


        if [ ! ${TICK_CPP_TEST_RESULT} -eq 0 ]
          then
            echo "C++ tests failed"
            exit ${TICK_CPP_TEST_RESULT}
        fi

        echo ""
        echo "PYTHON TEST"
        (${PYTHON_EXEC} -m unittest discover -v . "*_test.py")
        TICK_PYTHON_TEST_RESULT=$?

        if [ ! ${TICK_PYTHON_TEST_RESULT} -eq 0 ]
          then
            echo "Python tests failed"
            exit ${TICK_PYTHON_TEST_RESULT}
        fi
        echo "PYTHON TEST DONE"

        echo ""
        echo "CPPLINT STYLECHECK"
        (${PYTHON_EXEC} -c "import cpplint")
        if [ ! $? -eq 0 ]
          then
            echo "Stylecheck by cpplint failed because cpplint is not installed as a Python module"
        elif [ ! -z "$TICK_NO_CPPLINT" ]
          then
            echo "Stylecheck by cpplint explicitely disabled - skipped"
        else
            (${PYTHON_EXEC} -m cpplint --recursive "${CPPLINT_DIRS[@]}")
            TICK_CPPLINT_RESULT=$?

            if [ ! ${TICK_CPPLINT_RESULT} -eq 0 ]
              then
                echo "Codestyle check by cpplint failed"
                exit ${TICK_CPPLINT_RESULT}
            fi

            echo "CPPLINT STYLECHECK DONE"
        fi

        exit 0
fi
