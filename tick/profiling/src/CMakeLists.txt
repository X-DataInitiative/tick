add_executable(tick_profile_poisson_sdca poisson_sdca.cpp)

target_link_libraries(tick_profile_poisson_sdca
        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_MODEL}
        ${TICK_LIB_SOLVER}
        ${TICK_LIB_PROX}
        ${TICK_LIB_CRANDOM}
        ${TICK_BLAS_LIBRARIES}
        )

add_executable(tick_profile_hawkes_sdca hawkes_sdca.cpp)

target_link_libraries(tick_profile_hawkes_sdca
        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_MODEL}
        ${TICK_LIB_SOLVER}
        ${TICK_LIB_PROX}
        ${TICK_LIB_CRANDOM}
        ${TICK_LIB_INFERENCE}
        ${TICK_BLAS_LIBRARIES}
        ${TICK_LIB_SIMULATION}
        )