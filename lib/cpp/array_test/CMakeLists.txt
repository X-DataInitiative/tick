set(TICK_ARRAY_TEST_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/array_test)

add_library(tick_array_test EXCLUDE_FROM_ALL
        array_test.cpp 
        ${TICK_ARRAY_TEST_INCLUDE_DIR}/array_test.h
        varraycontainer.cpp 
        ${TICK_ARRAY_TEST_INCLUDE_DIR}/varraycontainer.h
        ${TICK_ARRAY_TEST_INCLUDE_DIR}/sbasearray_container.h
        sbasearray_container.cpp
        ${TICK_ARRAY_TEST_INCLUDE_DIR}/performance_test.h
        performance_test.cpp
        ${TICK_ARRAY_TEST_INCLUDE_DIR}/timer.h
        ${TICK_ARRAY_TEST_INCLUDE_DIR}/typemap_test.h
        typemap_test.cpp)
