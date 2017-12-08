add_executable(tick_test_linear_model models_gtest.cpp)

target_link_libraries(tick_test_linear_model
    ${TICK_LIB_ARRAY}
    ${TICK_LIB_BASE}
    ${TICK_LIB_BASE_MODEL}
    ${TICK_LIB_LINEAR_MODEL}
    ${TICK_TEST_LIBS}
    )
