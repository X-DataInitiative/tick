add_executable(tick_test_array array_gtest.cpp)
add_executable(tick_test_varray varray_gtest.cpp)

target_link_libraries(tick_test_array ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${EXTRA_TEST_LIBS} array)
target_link_libraries(tick_test_varray ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${EXTRA_TEST_LIBS} array)
