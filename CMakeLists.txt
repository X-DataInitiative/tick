cmake_minimum_required(VERSION 3.24)

project(tick LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE Python_NumPy_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT Python_NumPy_INCLUDE_DIR)
  message(FATAL_ERROR "Could not determine NumPy include directory")
endif()

set(TICK_ENABLE_FAST_MATH ON CACHE BOOL "Enable fast math flags")
set(TICK_BUILD_STAGE1_SPIKE ON CACHE BOOL "Build the pybind11 array/base spike")
set(TICK_REBUILD_LIBS ON CACHE BOOL "Rebuild the tick libraries via CMake" FORCE)
set(USE_MKL OFF CACHE BOOL "Force tick to use MKL" FORCE)
set(USE_BLAS OFF CACHE BOOL "Force tick to use BLAS" FORCE)
set(BENCHMARK OFF CACHE BOOL "Build C++ benchmarks" FORCE)

if(TICK_ENABLE_FAST_MATH)
  if(MSVC)
    add_compile_options(/fp:fast)
  else()
    add_compile_options(-ffast-math)
  endif()
endif()

add_compile_definitions(
  PYTHON_LINK
  TICK_REBUILD_LIBS=ON
  TICK_PYTHON_BINDINGS
)

if(WIN32 AND TICK_REBUILD_LIBS)
  add_compile_definitions(TICK_STATIC_LIBS)
endif()

if(APPLE)
  add_compile_definitions(DARWIN)
endif()

include_directories(
  "${PROJECT_SOURCE_DIR}/lib/include"
  "${PROJECT_SOURCE_DIR}/lib/third_party/cereal/include"
  "${Python_INCLUDE_DIRS}"
  "${Python_NumPy_INCLUDE_DIR}"
)

add_subdirectory(lib)

function(tick_pybind_module target_name output_name source_file install_dir)
  pybind11_add_module(${target_name} MODULE "${source_file}")
  target_link_libraries(${target_name} PRIVATE tick_array tick_base ${ARGN})
  target_include_directories(${target_name} PRIVATE
    "${PROJECT_SOURCE_DIR}/python/src"
    "${PROJECT_SOURCE_DIR}/lib/include"
    "${PROJECT_SOURCE_DIR}/lib/third_party/cereal/include"
    "${Python_INCLUDE_DIRS}"
    "${Python_NumPy_INCLUDE_DIR}"
  )
  set_target_properties(${target_name} PROPERTIES
    OUTPUT_NAME "${output_name}"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${install_dir}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${install_dir}"
  )
  install(TARGETS ${target_name}
    LIBRARY DESTINATION "${install_dir}"
    RUNTIME DESTINATION "${install_dir}"
  )
endfunction()

if(TICK_BUILD_STAGE1_SPIKE)
  tick_pybind_module(
    tick_array_py array python/src/tick_array_module.cpp tick/array/build
  )
  tick_pybind_module(
    tick_base_py base python/src/tick_base_module.cpp tick/base/build
  )
  tick_pybind_module(
    tick_array_test_py array_test python/src/tick_array_test_module.cpp
    tick/array_test/build tick_array_test
  )
  tick_pybind_module(
    tick_random_py crandom python/src/tick_random_module.cpp tick/random/build
    tick_crandom
  )
  tick_pybind_module(
    tick_base_model_py base_model python/src/tick_base_model_module.cpp
    tick/base_model/build tick_base_model
  )
  tick_pybind_module(
    tick_linear_model_py linear_model python/src/tick_linear_model_module.cpp
    tick/linear_model/build tick_linear_model tick_base_model
  )
  tick_pybind_module(
    tick_prox_py prox python/src/tick_prox_module.cpp tick/prox/build
    tick_prox
  )
  tick_pybind_module(
    tick_solver_py solver python/src/tick_solver_module.cpp tick/solver/build
    tick_solver tick_prox tick_base_model tick_linear_model tick_robust
    tick_survival tick_crandom
  )
  tick_pybind_module(
    tick_robust_py robust python/src/tick_robust_module.cpp tick/robust/build
    tick_robust tick_linear_model tick_base_model
  )
  tick_pybind_module(
    tick_preprocessing_py preprocessing
    python/src/tick_preprocessing_module.cpp tick/preprocessing/build
    tick_preprocessing
  )
  tick_pybind_module(
    tick_hawkes_simulation_py hawkes_simulation
    python/src/tick_hawkes_simulation_module.cpp
    tick/hawkes/simulation/build
    tick_hawkes_simulation tick_hawkes_model tick_crandom
  )
  tick_pybind_module(
    tick_hawkes_model_py hawkes_model
    python/src/tick_hawkes_model_module.cpp
    tick/hawkes/model/build
    tick_hawkes_model tick_base_model
  )
  tick_pybind_module(
    tick_hawkes_inference_py hawkes_inference
    python/src/tick_hawkes_inference_module.cpp
    tick/hawkes/inference/build
    tick_hawkes_inference tick_hawkes_model tick_base_model
  )
  tick_pybind_module(
    tick_survival_py survival python/src/tick_survival_module.cpp
    tick/survival/build
    tick_survival tick_base_model
  )
endif()
