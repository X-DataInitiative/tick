set(TICK_HAWKES_INFERENCE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/hawkes/inference)

add_library(tick_hawkes_inference EXCLUDE_FROM_ALL
        ${TICK_HAWKES_INFERENCE_INCLUDE_DIR}/hawkes_conditional_law.h
        ${TICK_HAWKES_INFERENCE_INCLUDE_DIR}/hawkes_em.h
        ${TICK_HAWKES_INFERENCE_INCLUDE_DIR}/hawkes_adm4.h
        ${TICK_HAWKES_INFERENCE_INCLUDE_DIR}/hawkes_basis_kernels.h
        ${TICK_HAWKES_INFERENCE_INCLUDE_DIR}/hawkes_sumgaussians.h
        hawkes_adm4.cpp
        hawkes_basis_kernels.cpp
        hawkes_conditional_law.cpp
        hawkes_em.cpp
        hawkes_sumgaussians.cpp
        )

target_link_libraries(tick_hawkes_inference
        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_CRANDOM})