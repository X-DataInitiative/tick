add_executable(tick_test_solver svrg_gtest.cpp)

target_link_libraries(tick_test_solver
    ${TICK_LIB_ARRAY}
    ${TICK_LIB_BASE}
    ${TICK_LIB_BASE_MODEL}
    ${TICK_LIB_LINEAR_MODEL}
    ${TICK_LIB_MODEL}
    ${TICK_LIB_PROX}
    ${TICK_LIB_CRANDOM}
    ${TICK_LIB_SOLVER}
    ${TICK_TEST_LIBS}
    )
