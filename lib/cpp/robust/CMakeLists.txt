set(TICK_ROBUST_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/robust)

add_library(tick_robust EXCLUDE_FROM_ALL
        ${TICK_ROBUST_INCLUDE_DIR}/model_absolute_regression.h
        ${TICK_ROBUST_INCLUDE_DIR}/model_epsilon_insensitive.h
        ${TICK_ROBUST_INCLUDE_DIR}/model_epsilon_insensitive.h
        ${TICK_ROBUST_INCLUDE_DIR}/model_huber.h
        ${TICK_ROBUST_INCLUDE_DIR}/model_modified_huber.h
        ${TICK_ROBUST_INCLUDE_DIR}/model_linreg_with_intercepts.h
        model_absolute_regression.cpp
        model_epsilon_insensitive.cpp
        model_epsilon_insensitive.cpp
        model_huber.cpp
        model_modified_huber.cpp
        model_linreg_with_intercepts.cpp
        )