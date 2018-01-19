add_executable(tick_test_hawkes_model
        hawkes_models_gtest.cpp
        )

target_link_libraries(tick_test_hawkes_model
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_BASE}
        ${TICK_LIB_BASE_MODEL}
        ${TICK_LIB_HAWKES_MODEL}
        ${TICK_TEST_LIBS}
        )


