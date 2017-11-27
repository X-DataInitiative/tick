set(TICK_INFERENCE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/inference)

add_library(tick_inference EXCLUDE_FROM_ALL
        ${TICK_INFERENCE_INCLUDE_DIR}/hawkes_conditional_law.h
        hawkes_conditional_law.cpp
        hawkes_em.cpp 
        ${TICK_INFERENCE_INCLUDE_DIR}/hawkes_em.h
        ${TICK_INFERENCE_INCLUDE_DIR}/hawkes_adm4.h
        hawkes_adm4.cpp
        hawkes_basis_kernels.cpp 
        ${TICK_INFERENCE_INCLUDE_DIR}/hawkes_basis_kernels.h
        ${TICK_INFERENCE_INCLUDE_DIR}/hawkes_sumgaussians.h
        hawkes_sumgaussians.cpp)

target_link_libraries(tick_inference

        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_MODEL})
