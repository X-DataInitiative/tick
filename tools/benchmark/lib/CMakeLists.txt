add_executable(tick_saga_sparse saga_sparse.cpp)
target_link_libraries(tick_saga_sparse
        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_CRANDOM}
        ${TICK_LIB_BASE_MODEL}
        ${TICK_LIB_LINEAR_MODEL}
        ${TICK_LIB_PROX}
        ${TICK_LIB_SOLVER}
        ${TICK_TEST_LIBS}
        )

add_executable(tick_asaga_sparse asaga_sparse.cpp)
target_link_libraries(tick_asaga_sparse
        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_CRANDOM}
        ${TICK_LIB_BASE_MODEL}
        ${TICK_LIB_LINEAR_MODEL}
        ${TICK_LIB_PROX}
        ${TICK_LIB_SOLVER}
        ${TICK_TEST_LIBS}
        )


add_executable(tick_hawkes_least_squares_weights hawkes_least_squares_weights.cpp)
target_link_libraries(tick_hawkes_least_squares_weights
        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_CRANDOM}
        ${TICK_LIB_BASE_MODEL}
        ${TICK_LIB_HAWKES_MODEL}
        ${TICK_LIB_PROX}
        ${TICK_LIB_SOLVER}
        )


add_executable(tick_matrix_vector_product matrix_vector_product.cpp)
target_link_libraries(tick_matrix_vector_product
        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_CRANDOM}
        )

add_executable(tick_logistic_regression_loss logistic_regression_loss.cpp)
target_link_libraries(tick_logistic_regression_loss
        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_CRANDOM}
        ${TICK_LIB_BASE_MODEL}
        ${TICK_LIB_LINEAR_MODEL}
        )

