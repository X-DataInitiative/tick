add_executable(tick_poisson_sdca poisson_sdca.cpp)

target_link_libraries(tick_poisson_sdca

        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_MODEL}
        ${TICK_LIB_SOLVER}
        ${TICK_LIB_PROX}
        ${TICK_LIB_CRANDOM}
        ${TICK_BLAS_LIBRARIES}
        )

add_executable(tick_sgd_random_order sgd_random_order.cpp)

target_link_libraries(tick_sgd_random_order

        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_MODEL}
        ${TICK_LIB_SOLVER}
        ${TICK_LIB_PROX}
        ${TICK_LIB_CRANDOM}
        ${TICK_BLAS_LIBRARIES}
        )

add_executable(tick_logreg_grad model_logreg_grad.cpp)
target_link_libraries(tick_logreg_grad
        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_MODEL}
        ${TICK_LIB_SOLVER}
        ${TICK_LIB_PROX}
        ${TICK_LIB_CRANDOM}
        ${TICK_BLAS_LIBRARIES}
        )


add_executable(tick_hawkes_least_sq hawkes_least_sq.cpp)
target_link_libraries(tick_hawkes_least_sq
        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_MODEL}
        ${TICK_LIB_SOLVER}
        ${TICK_LIB_PROX}
        ${TICK_LIB_CRANDOM}
        ${TICK_BLAS_LIBRARIES}
        )