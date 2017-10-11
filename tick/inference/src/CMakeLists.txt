add_library(tick_inference EXCLUDE_FROM_ALL
        hawkes_conditional_law.h hawkes_conditional_law.cpp
        hawkes_em.cpp hawkes_em.h
        hawkes_adm4.h hawkes_adm4.cpp
        hawkes_basis_kernels.cpp hawkes_basis_kernels.h
        hawkes_sumgaussians.h hawkes_sumgaussians.cpp
        online_forest.cpp online_forest.h)

target_link_libraries(tick_inference

        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_MODEL})
