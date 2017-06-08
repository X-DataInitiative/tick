add_executable(tick_test_base utils_gtest.cpp)

target_link_libraries(tick_test_base ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${EXTRA_TEST_LIBS} array base )
