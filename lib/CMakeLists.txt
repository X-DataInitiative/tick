cmake_minimum_required(VERSION 3.7)
project(tick)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ffast-math")

include_directories(${CMAKE_SOURCE_DIR}/..)

option(USE_MKL "Force tick to use MKL" OFF)
option(USE_BLAS "Force tick to use BLAS" OFF)
option(BENCHMARK "Build benchmarks" OFF)

set(TICK_EXTRA_RPATH "")
set(TICK_BLAS_LIBRARIES "")

if (${USE_MKL})
    list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
    find_package(MKL REQUIRED)
    message(STATUS "Using MKL")

    add_definitions(-DTICK_USE_MKL)

    include_directories(${MKL_INCLUDE_DIRECTORIES})

    set(TICK_BLAS_LIBRARIES ${MKL_LIBRARIES} ${INTEL_LIBRARIES})
    set(TICK_EXTRA_RPATH "${TICK_EXTRA_RPATH};${INTEL_LIB};${MKL_LIB}")

    message(STATUS "MKL include directory: " ${MKL_INCLUDE_DIRECTORIES})

elseif (${USE_BLAS})
    find_package(BLAS REQUIRED)
    message(STATUS "Using BLAS")

    set(TICK_BLAS_LIBRARIES ${BLAS_LIBRARIES})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTICK_USE_CBLAS")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")

    message(STATUS "BLAS Libraries: " ${BLAS_LIBRARIES})
    message(STATUS "BLAS Linker flags: " ${BLAS_LINKER_FLAGS})
else ()
    message(STATUS "Using no BLAS nor MKL")
endif ()

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/cereal/include)

option(TICK_REBUILD_LIBS "If enabled, rebuild the tick libraries via CMake instead of linking to the Python extensions" ON)

if (${TICK_REBUILD_LIBS})
    message(STATUS "Rebuilding tick libraries")
    set(TICK_LIB_BASE tick_base)
    set(TICK_LIB_ARRAY tick_array)
    set(TICK_LIB_BASE_MODEL tick_base_model)
    set(TICK_LIB_LINEAR_MODEL tick_linear_model)
    set(TICK_LIB_CRANDOM tick_crandom)
    set(TICK_LIB_ARRAY_TEST tick_array_test)
    set(TICK_LIB_HAWKES_INFERENCE tick_hawkes_inference)
    set(TICK_LIB_HAWKES_MODEL tick_hawkes_model)
    set(TICK_LIB_HAWKES_SIMULATION tick_hawkes_simulation)
    set(TICK_LIB_PREPROCESSING tick_preprocessing)
    set(TICK_LIB_PROX tick_prox)
    set(TICK_LIB_ROBUST tick_robust)
    set(TICK_LIB_SOLVER tick_solver)
    set(TICK_LIB_SURVIVAL tick_survival)
endif ()

add_subdirectory(cpp/array)
add_subdirectory(cpp/base)
add_subdirectory(cpp/array_test)
add_subdirectory(cpp/random)
add_subdirectory(cpp/base_model)
add_subdirectory(cpp/linear_model)
add_subdirectory(cpp/hawkes/inference)
add_subdirectory(cpp/hawkes/model)
add_subdirectory(cpp/hawkes/simulation)
add_subdirectory(cpp/prox)
add_subdirectory(cpp/solver)
add_subdirectory(cpp/robust)
add_subdirectory(cpp/survival)
add_subdirectory(cpp/preprocessing)

if (APPLE)
    set(TICK_INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../tick/array/build;${CMAKE_CURRENT_SOURCE_DIR}/../tick/base/build;${CMAKE_CURRENT_SOURCE_DIR}/../tick/base_model/build;${CMAKE_CURRENT_SOURCE_DIR}/../tick/linear_model/build;${CMAKE_CURRENT_SOURCE_DIR}/../tick/random/build;${CMAKE_CURRENT_SOURCE_DIR}/../tick/prox/build;${CMAKE_CURRENT_SOURCE_DIR}/../tick/solver/build;${CMAKE_CURRENT_SOURCE_DIR}/../tick/hawkes/inference/build;${CMAKE_CURRENT_SOURCE_DIR}/../tick/hawkes/model/build;${CMAKE_CURRENT_SOURCE_DIR}/../tick/hawkes/simulation/build;${CMAKE_CURRENT_SOURCE_DIR}/../tick/robust/build;${CMAKE_CURRENT_SOURCE_DIR}/../tick/survival/build;${TICK_EXTRA_RPATH}")

    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH "${TICK_INSTALL_RPATH}")
endif (APPLE)

find_package(GTest)

if (${GTEST_FOUND})
    enable_testing()
    message(STATUS "Testing in C++ enabled")

    find_package(Threads REQUIRED)
    find_package(PythonInterp 3.7 REQUIRED)
    find_package(PythonLibs 3.7 REQUIRED)

    include_directories(${GTEST_INCLUDE_DIRS})

    set(TICK_TEST_LIBS
            ${PYTHON_LIBRARY}
            ${GTEST_BOTH_LIBRARIES}
            ${CMAKE_THREAD_LIBS_INIT}
            ${TICK_BLAS_LIBRARIES} pthread)

    if (APPLE)
        find_library(ACCELERATE_LIBRARY Accelerate)

        mark_as_advanced(ACCELERATE_LIBRARY)
        set(TICK_TEST_LIBS ${TICK_TEST_LIBS} ${ACCELERATE_LIBRARY})

    else (APPLE)
        set(TICK_TEST_LIBS ${TICK_TEST_LIBS} -lrt -ldl -lutil)
    endif (APPLE)

    add_subdirectory(cpp-test/base)
    add_subdirectory(cpp-test/array)
    add_subdirectory(cpp-test/hawkes/model)
    add_subdirectory(cpp-test/hawkes/simulation)
    add_subdirectory(cpp-test/linear_model)
    add_subdirectory(cpp-test/solver)

    add_custom_target(check
            COMMAND cpp-test/base/tick_test_base
            COMMAND cpp-test/array/tick_test_array
            COMMAND cpp-test/array/tick_test_varray
            COMMAND cpp-test/linear_model/tick_test_linear_model
            COMMAND cpp-test/hawkes/model/tick_test_hawkes_model
            COMMAND cpp-test/hawkes/simulation/tick_test_hawkes_simulation
            COMMAND cpp-test/solver/tick_test_svrg
            )

else ()
    message(STATUS "C++ Testing NOT enabled")
    add_custom_target(
      check COMMAND echo "Could not run C++ checks - gtest framework not found"
    )
endif ()

if (${BENCHMARK})
    find_package(Threads REQUIRED)
    set(TICK_TEST_LIBS
        ${PYTHON_LIBRARY}
        ${CMAKE_THREAD_LIBS_INIT}
        ${TICK_BLAS_LIBRARIES} pthread
    )
    if (APPLE)
        find_library(ACCELERATE_LIBRARY Accelerate)

        mark_as_advanced(ACCELERATE_LIBRARY)
        set(TICK_TEST_LIBS ${TICK_TEST_LIBS} ${ACCELERATE_LIBRARY})

    else (APPLE)
        set(TICK_TEST_LIBS ${TICK_TEST_LIBS} -lrt -ldl -lutil -pthread)
    endif (APPLE)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../tools/benchmark/lib benchmarks)
    add_custom_target(benchmarks
            COMMAND benchmarks/tick_saga_sparse
            COMMAND benchmarks/tick_asaga_sparse
            COMMAND benchmarks/tick_hawkes_least_squares_weights
            COMMAND benchmarks/tick_matrix_vector_product
            COMMAND benchmarks/tick_logistic_regression_loss
            )
else ()
    message(STATUS "C++ benchmarking NOT enabled")
endif ()
