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
        set(TICK_INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/base/build;${CMAKE_CURRENT_SOURCE_DIR}/base/array/build;${CMAKE_CURRENT_SOURCE_DIR}/optim/model/build;${CMAKE_CURRENT_SOURCE_DIR}/random/build;${CMAKE_CURRENT_SOURCE_DIR}/optim/prox/build;${CMAKE_CURRENT_SOURCE_DIR}/optim/solver/build;${CMAKE_CURRENT_SOURCE_DIR}/simulation/build")

        set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
        set(CMAKE_INSTALL_RPATH "${TICK_INSTALL_RPATH}")

        find_library(ACCELERATE_LIBRARY Accelerate)
        mark_as_advanced(ACCELERATE_LIBRARY)
        set(TICK_TEST_LIBS ${TICK_TEST_LIBS} ${ACCELERATE_LIBRARY})
    else(APPLE)
        set(TICK_TEST_LIBS ${TICK_TEST_LIBS} -lrt -ldl -lutil)
    endif(APPLE)

    add_subdirectory(base/tests/src)
    add_subdirectory(base/array/tests/src)
    add_subdirectory(optim/model/tests/src)
    add_subdirectory(optim/solver/tests/src)
    add_subdirectory(simulation/tests/src)
    add_subdirectory(inference/tests/src)

    add_custom_target(check
            COMMAND base/tests/src/tick_test_base
            COMMAND base/array/tests/src/tick_test_array
            COMMAND base/array/tests/src/tick_test_varray
            COMMAND optim/model/tests/src/tick_test_model
            COMMAND simulation/tests/src/tick_test_hawkes
            COMMAND optim/solver/tests/src/tick_test_solver
            COMMAND inference/tests/src/tick_test_inference
            )
else()
    message(STATUS "Testing in C++ NOT enabled")

    add_custom_target(check COMMAND
            echo "Could not run C++ checks - gtest framework not found"
            )
endif()
