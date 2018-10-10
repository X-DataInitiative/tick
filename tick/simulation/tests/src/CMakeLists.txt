add_executable(tick_test_hawkes
        hawkes_kernel_exp_gtest.cpp
        hawkes_kernel_0_gtest.cpp
        hawkes_kernel_power_law_gtest.cpp
        hawkes_kernel_time_func_gtest.cpp
        hawkes_kernel_sumexp_gtest.cpp
        hawkes_simulation.cpp
        hawkes_custom_test.cpp
        hawkes_kernel_sumexp_lag_gtest.cpp)

target_link_libraries(tick_test_hawkes
    ${TICK_LIB_ARRAY}
    ${TICK_LIB_BASE}
    ${TICK_LIB_CRANDOM}
    ${TICK_LIB_SIMULATION}

    ${TICK_TEST_LIBS})


