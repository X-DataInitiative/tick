add_executable(tick_test_array array_gtest.cpp)
add_executable(tick_test_varray varray_gtest.cpp)

target_link_libraries(tick_test_array
        ${TICK_TEST_LIBS}
        ${TICK_LIB_ARRAY}
    )

target_link_libraries(tick_test_varray
        ${TICK_TEST_LIBS}
        ${TICK_LIB_ARRAY}
    )
