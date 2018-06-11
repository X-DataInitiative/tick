add_executable(tick_test_array array_gtest.cpp)

target_link_libraries(tick_test_array
        ${TICK_LIB_ARRAY}
        ${TICK_TEST_LIBS}
    )

add_executable(tick_test_atomic_array atomic_array_gtest.cpp)

target_link_libraries(tick_test_atomic_array
        ${TICK_LIB_ARRAY}
        ${TICK_TEST_LIBS}
        )

add_executable(tick_test_varray varray_gtest.cpp)

target_link_libraries(tick_test_varray
        ${TICK_LIB_ARRAY}
        ${TICK_TEST_LIBS}
    )

add_executable(tick_test_serializer serializer_gtest.cpp)

target_link_libraries(tick_test_serializer
        ${TICK_LIB_ARRAY}
        ${TICK_TEST_LIBS}
        )
