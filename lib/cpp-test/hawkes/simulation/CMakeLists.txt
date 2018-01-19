add_executable(tick_test_hawkes_simulation
        hawkes_kernel_exp_gtest.cpp
        hawkes_kernel_0_gtest.cpp
        hawkes_kernel_power_law_gtest.cpp
        hawkes_kernel_time_func_gtest.cpp
        hawkes_kernel_sumexp_gtest.cpp
        hawkes_simulation.cpp
        )

target_link_libraries(tick_test_hawkes_simulation
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_BASE}
        ${TICK_LIB_CRANDOM}
        ${TICK_LIB_HAWKES_SIMULATION}
        ${TICK_LIB_HAWKES_MODEL}
        ${TICK_TEST_LIBS}
        )


