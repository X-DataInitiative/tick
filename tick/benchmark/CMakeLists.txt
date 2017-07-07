project(linreg_perf_test)

include_directories(${CMAKE_SOURCE_DIR}/..)

add_executable(benchmark_test linreg_perf.cpp)

set(BENCHMARK_LIBS model ${EXTRA_TEST_LIBS})

target_link_libraries(benchmark_test ${BENCHMARK_LIBS})
