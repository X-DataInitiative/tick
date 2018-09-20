add_executable(tick_poisson_sdca poisson_sdca.cpp)

target_link_libraries(tick_poisson_sdca

        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_LINEAR_MODEL}
        ${TICK_LIB_SOLVER}
        ${TICK_LIB_PROX}
        ${TICK_LIB_CRANDOM}
        ${TICK_BLAS_LIBRARIES}
        ${TICK_TEST_LIBS}
        )

add_executable(tick_logreg_grad model_logreg_grad.cpp)
target_link_libraries(tick_logreg_grad
        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_LINEAR_MODEL}
        ${TICK_LIB_SOLVER}
        ${TICK_LIB_PROX}
        ${TICK_LIB_CRANDOM}
        ${TICK_BLAS_LIBRARIES}
        ${TICK_TEST_LIBS}
        )


add_executable(tick_hawkes_least_sq hawkes_least_sq.cpp)
target_link_libraries(tick_hawkes_least_sq
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_CRANDOM}
        ${TICK_LIB_LINEAR_MODEL}
        ${TICK_LIB_PROX}
        ${TICK_LIB_SOLVER}
        ${TICK_BLAS_LIBRARIES}
        ${TICK_LIB_HAWKES_SIMULATION}
        ${TICK_LIB_HAWKES_MODEL}
        ${TICK_TEST_LIBS}
        )



add_executable(matrix_vector_dotproduct matrix_vector_dotproduct.cpp)

target_link_libraries(
        matrix_vector_dotproduct

        ${TICK_LIB_ARRAY}
        ${TICK_LIB_BASE}
        ${TICK_LIB_CRANDOM}
        ${TICK_LIB_LINEAR_MODEL}

        ${TICK_TEST_LIBS}
)

add_executable(hawkes_leastsq_weights hawkes_leastsq_weights.cpp)

target_link_libraries(
        hawkes_leastsq_weights

        ${TICK_LIB_ARRAY}
        ${TICK_LIB_BASE}
        ${TICK_LIB_CRANDOM}
        ${TICK_LIB_BASE_MODEL}
        ${TICK_LIB_HAWKES_MODEL}
        ${TICK_LIB_HAWKES_SIMULATION}
        ${TICK_LIB_PROX}
        ${TICK_LIB_SOLVER}
        ${TICK_TEST_LIBS}
)
#
add_executable(logistic_regression_loss logistic_regression_loss.cpp)

target_link_libraries(
        logistic_regression_loss

        ${TICK_LIB_ARRAY}
        ${TICK_LIB_BASE}
        ${TICK_LIB_BASE_MODEL}
        ${TICK_LIB_LINEAR_MODEL}
        ${TICK_LIB_CRANDOM}

        ${TICK_TEST_LIBS}
)
