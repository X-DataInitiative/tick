cmake_minimum_required(VERSION 2.8)
project(tick)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ffast-math")
set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(BLAS)

set(TICK_BLAS_LIBRARIES "")

if (${BLAS_FOUND})
    message(STATUS "Using BLAS")
    add_definitions(-DTICK_CBLAS_AVAILABLE)

    set(TICK_BLAS_LIBRARIES ${BLAS_LIBRARIES})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")

    message(STATUS "BLAS Libraries: " ${BLAS_LIBRARIES})
    message(STATUS "BLAS Linker flags: " ${BLAS_LINKER_FLAGS})
endif()

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/cereal/include)

option(TICK_REBUILD_LIBS "If enabled, rebuild the tick libraries via CMake instead of linking to the Python extensions" ON)

if (${TICK_REBUILD_LIBS})
    message(STATUS "Rebuilding tick libraries")

    set(TICK_LIB_BASE       tick_base)
    set(TICK_LIB_ARRAY      tick_array)
    set(TICK_LIB_MODEL      tick_model)
    set(TICK_LIB_SIMULATION tick_simulation)
    set(TICK_LIB_CRANDOM    tick_crandom)
    set(TICK_LIB_ARRAY_TEST tick_array_test)
    set(TICK_LIB_INFERENCE  tick_inference)
    set(TICK_LIB_PREPROCESSING tick_preprocessing)
    set(TICK_LIB_PROX       tick_prox)
    set(TICK_LIB_SOLVER     tick_solver)
endif()


add_subdirectory(src/cpp/base)
add_subdirectory(src/cpp/array)
add_subdirectory(src/cpp/array_test)
add_subdirectory(src/cpp/random)
add_subdirectory(src/cpp/simulation)
add_subdirectory(src/cpp/optim/model)
add_subdirectory(src/cpp/optim/prox)
add_subdirectory(src/cpp/optim/solver)
add_subdirectory(src/cpp/inference)
add_subdirectory(src/cpp/preprocessing)

find_package(GTest)

if (${GTEST_FOUND})
    enable_testing()
    message(STATUS "Testing in C++ enabled")

    find_package(Threads REQUIRED)
    find_package(PythonInterp REQUIRED)
    find_package(PythonLibs REQUIRED)

    include_directories(${GTEST_INCLUDE_DIRS})

    set(TICK_TEST_LIBS
        ${PYTHON_LIBRARIES}
        ${GTEST_BOTH_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ${TICK_BLAS_LIBRARIES})

    if(APPLE)
        set(TICK_INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../tick/base/build;${CMAKE_CURRENT_SOURCE_DIR}/../tick/array/build;${CMAKE_CURRENT_SOURCE_DIR}/../tick/optim/model/build;${CMAKE_CURRENT_SOURCE_DIR}/../tick/random/build;${CMAKE_CURRENT_SOURCE_DIR}/../tick/optim/prox/build;${CMAKE_CURRENT_SOURCE_DIR}/../tick/optim/solver/build;${CMAKE_CURRENT_SOURCE_DIR}/../tick/simulation/build")

        set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
        set(CMAKE_INSTALL_RPATH "${TICK_INSTALL_RPATH}")

        find_library(ACCELERATE_LIBRARY Accelerate)
        mark_as_advanced(ACCELERATE_LIBRARY)
        set(TICK_TEST_LIBS ${TICK_TEST_LIBS} ${ACCELERATE_LIBRARY})
    else(APPLE)
        set(TICK_TEST_LIBS ${TICK_TEST_LIBS} -lrt -ldl -lutil)
    endif(APPLE)

    add_subdirectory(src/cpp-test/base)
    add_subdirectory(src/cpp-test/array)
    add_subdirectory(src/cpp-test/optim/model)
    add_subdirectory(src/cpp-test/optim/solver)
    add_subdirectory(src/cpp-test/simulation)

    add_custom_target(check
            COMMAND src/cpp-test/base/tick_test_base
            COMMAND src/cpp-test/array/tick_test_array
            COMMAND src/cpp-test/array/tick_test_varray
            COMMAND src/cpp-test/optim/model/tick_test_model
            COMMAND src/cpp-test/simulation/tick_test_hawkes
            COMMAND src/cpp-test/optim/solver/tick_test_solver
            )
else()
    message(STATUS "Testing in C++ NOT enabled")

    add_custom_target(check COMMAND
            echo "Could not run C++ checks - gtest framework not found"
            )
endif()
