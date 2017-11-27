set(TICK_RANDOM_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/random)

add_library(tick_crandom EXCLUDE_FROM_ALL
        rand.cpp 
        ${TICK_RANDOM_INCLUDE_DIR}/rand.h
        test_rand.cpp 
        ${TICK_RANDOM_INCLUDE_DIR}/test_rand.h)
