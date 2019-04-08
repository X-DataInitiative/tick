set(TICK_SURVIVAL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/survival)

add_library(tick_survival EXCLUDE_FROM_ALL
        ${TICK_SURVIVAL_INCLUDE_DIR}/model_coxreg_partial_lik.h
        ${TICK_SURVIVAL_INCLUDE_DIR}/model_sccs.h
        model_coxreg_partial_lik.cpp
        model_sccs.cpp
        )
