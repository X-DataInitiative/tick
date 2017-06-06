add_executable(tick_test_model models_gtest.cpp hawkes_models_gtests.cpp)

target_link_libraries(tick_test_model ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${EXTRA_TEST_LIBS} base array model)
