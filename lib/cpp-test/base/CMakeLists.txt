add_executable(tick_test_base utils_gtest.cpp)

target_link_libraries(tick_test_base
    ${TICK_LIB_ARRAY}
    ${TICK_LIB_BASE}

    ${TICK_TEST_LIBS}
    )
