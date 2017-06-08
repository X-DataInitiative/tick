add_library(inference EXCLUDE_FROM_ALL
        hawkes_conditional_law.h hawkes_conditional_law.cpp
        hawkes_em.cpp hawkes_em.h
        hawkes_adm4.h hawkes_adm4.cpp
        hawkes_basis_kernels.cpp hawkes_basis_kernels.h
        hawkes_sumgaussians.h hawkes_sumgaussians.cpp)

target_link_libraries(inference base array model)
