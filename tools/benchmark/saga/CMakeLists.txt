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
