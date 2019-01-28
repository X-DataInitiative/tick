#include_directories(${CMAKE_SOURCE_DIR}/../include)

add_executable(matrix_vector_dotproduct matrix_vector_dotproduct.cpp)

target_link_libraries(
        matrix_vector_dotproduct

        ${TICK_LIB_ARRAY}
        ${TICK_LIB_BASE}
        ${TICK_LIB_CRANDOM}

        ${TICK_TEST_LIBS}
)

add_executable(hawkes_leastsq_weights hawkes_leastsq_weights.cpp)

target_link_libraries(
        hawkes_leastsq_weights

        ${TICK_LIB_MODEL}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_BASE}
        ${TICK_LIB_BASE_MODEL}
        ${TICK_LIB_LINEAR_MODEL}
        ${TICK_LIB_HAWKES_MODEL}
        ${TICK_LIB_CRANDOM}
        ${TICK_LIB_PROX}
        ${TICK_LIB_SOLVER}
        ${TICK_TEST_LIBS}
)
#
add_executable(logistic_regression_loss logistic_regression_loss.cpp)

target_link_libraries(
        logistic_regression_loss

        ${TICK_LIB_MODEL}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_BASE}
        ${TICK_LIB_BASE_MODEL}
        ${TICK_LIB_LINEAR_MODEL}
        ${TICK_LIB_CRANDOM}

        ${TICK_TEST_LIBS}
)
