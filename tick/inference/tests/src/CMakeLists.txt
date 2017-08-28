add_executable(tick_test_inference hawkes_sdca_test.cpp)

target_link_libraries(tick_test_inference
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_BASE}
        ${TICK_LIB_MODEL}
        ${TICK_LIB_SOLVER}
        ${TICK_LIB_INFERENCE}
        ${TICK_TEST_LIBS}
        )
