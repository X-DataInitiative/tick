set(TICK_SURVIVAL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/survival)

add_library(tick_survival EXCLUDE_FROM_ALL
        ${TICK_SURVIVAL_INCLUDE_DIR}/coxreg_partial_lik.h
        ${TICK_SURVIVAL_INCLUDE_DIR}/sccs.h
        coxreg_partial_lik.cpp
        sccs.cpp
        )