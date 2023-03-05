add_executable(tick_test_hawkes_inference
        hawkes_em_gtest.cpp
        )

target_link_libraries(tick_test_hawkes_inference
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_BASE}
        ${TICK_LIB_CRANDOM}
        ${TICK_LIB_BASE_MODEL}
        ${TICK_LIB_LINEAR_MODEL}
	${TICK_LIB_HAWKES_INFERENCE}
        ${TICK_LIB_HAWKES_MODEL}
        ${TICK_TEST_LIBS}
        )


