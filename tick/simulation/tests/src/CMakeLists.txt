add_executable(tick_test_hawkes
        hawkes_kernel_exp_gtest.cpp
        hawkes_kernel_0_gtest.cpp
        hawkes_kernel_power_law_gtest.cpp
        hawkes_kernel_time_func_gtest.cpp
        hawkes_kernel_sumexp_gtest.cpp
        hawkes_simulation.cpp
        )

target_link_libraries(tick_test_hawkes ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT}
        ${EXTRA_TEST_LIBS} base random simulation)
