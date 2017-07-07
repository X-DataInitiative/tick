cmake_minimum_required(VERSION 2.8)
project(tick)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
set(CMAKE_BUILD_TYPE Release)

option(USE_MKL "Force tick to use MKL" OFF)
option(USE_OPENMP "Force tick to use OpenMP" OFF)
option(USE_BLAS "Force tick to use BLAS" OFF)

if (${USE_MKL})
    set(MKLROOT $ENV{MKLROOT})
    include_directories(${MKLROOT}/include)

    set(EXTRA_TEST_LIBS mkl_intel_lp64 mkl_gnu_thread mkl_core gomp pthread m dl)

    add_definitions(-DTICK_USE_MKL)
elseif(${USE_BLAS})
    find_package(BLAS)

    if (${BLAS_FOUND})
        add_definitions(-DTICK_CBLAS_AVAILABLE)

        set(EXTRA_TEST_LIBS ${BLAS_LIBRARIES})
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")

        message(STATUS "BLAS Libraries: " ${BLAS_LIBRARIES})
        message(STATUS "BLAS Linker flags: " ${BLAS_LINKER_FLAGS})
    endif()
else()
endif()

if (${USE_OPENMP})
    find_package(OpenMP)

    if (${OPENMP_FOUND})
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

        add_definitions(-DTICK_USE_OPENMP)
    endif()
endif()

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/base/src
        ${CMAKE_CURRENT_SOURCE_DIR}/base/array/src
        ${CMAKE_CURRENT_SOURCE_DIR}/base/array_test/src
        ${CMAKE_CURRENT_SOURCE_DIR}/random/src
        ${CMAKE_CURRENT_SOURCE_DIR}/simulation/src
        ${CMAKE_CURRENT_SOURCE_DIR}/optim/model/src
        ${CMAKE_CURRENT_SOURCE_DIR}/optim/prox/src
        ${CMAKE_CURRENT_SOURCE_DIR}/optim/solver/src
        ${CMAKE_CURRENT_SOURCE_DIR}/inference/src
        ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/src
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/cereal/include)

add_subdirectory(base/src)
add_subdirectory(base/array/src)
add_subdirectory(base/array_test/src)
add_subdirectory(random/src)
add_subdirectory(simulation/src)
add_subdirectory(optim/model/src)
add_subdirectory(optim/prox/src)
add_subdirectory(optim/solver/src)
add_subdirectory(inference/src)
add_subdirectory(preprocessing/src)

set(EXTRA_LIBS ${EXTRA_LIBS} array base random array_test simulation model prox
        solver inference)

find_package(GTest)

if (${GTEST_FOUND})
    enable_testing()
    message(STATUS "Testing in C++ enabled")

    find_package (Threads REQUIRED)

    include_directories(${GTEST_INCLUDE_DIRS})

    if(APPLE)
        find_library(ACCELERATE_LIBRARY Accelerate)
        mark_as_advanced(ACCELERATE_LIBRARY)
        set(EXTRA_TEST_LIBS ${EXTRA_TEST_LIBS} ${ACCELERATE_LIBRARY})
    endif(APPLE)

    add_subdirectory(base/tests/src)
    add_subdirectory(base/array/tests/src)
    add_subdirectory(optim/model/tests/src)
    add_subdirectory(simulation/tests/src)

    add_custom_target(check
            COMMAND base/tests/src/tick_test_base
            COMMAND base/array/tests/src/tick_test_array
            COMMAND base/array/tests/src/tick_test_varray
            COMMAND optim/model/tests/src/tick_test_model
            COMMAND simulation/tests/src/tick_test_hawkes
            )
else()
    message(STATUS "Testing in C++ NOT enabled")

    add_custom_target(check COMMAND
            echo "Could not run C++ checks - gtest framework not found"
            )
endif()

add_subdirectory(benchmark)
