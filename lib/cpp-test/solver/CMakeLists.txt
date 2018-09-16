

add_executable(tick_test_saga saga_gtest.cpp)
target_link_libraries(tick_test_saga
    ${TICK_LIB_ARRAY}
    ${TICK_LIB_BASE}
    ${TICK_LIB_BASE_MODEL}
    ${TICK_LIB_CRANDOM}
    ${TICK_LIB_PROX}
    ${TICK_LIB_LINEAR_MODEL}
    ${TICK_LIB_ROBUST}
    ${TICK_LIB_SOLVER}
    ${TICK_TEST_LIBS}
)

add_executable(tick_test_svrg svrg_gtest.cpp)
target_link_libraries(tick_test_svrg
    ${TICK_LIB_ARRAY}
    ${TICK_LIB_BASE}
    ${TICK_LIB_BASE_MODEL}
    ${TICK_LIB_CRANDOM}
    ${TICK_LIB_PROX}
    ${TICK_LIB_LINEAR_MODEL}
    ${TICK_LIB_ROBUST}
    ${TICK_LIB_SOLVER}
    ${TICK_TEST_LIBS}
)
