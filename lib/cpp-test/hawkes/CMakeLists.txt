add_executable(tick_test_hawkes
    simulation/hawkes_kernel_exp_gtest.cpp
    simulation/hawkes_kernel_0_gtest.cpp
    simulation/hawkes_kernel_power_law_gtest.cpp
    simulation/hawkes_kernel_time_func_gtest.cpp
    simulation/hawkes_kernel_sumexp_gtest.cpp
    simulation/hawkes_simulation.cpp
)

target_link_libraries(tick_test_hawkes
    ${TICK_LIB_ARRAY}
    ${TICK_LIB_BASE}
    ${TICK_LIB_BASE_MODEL}
    ${TICK_LIB_CRANDOM}
    ${TICK_LIB_HAWKES}
    ${TICK_TEST_LIBS})


