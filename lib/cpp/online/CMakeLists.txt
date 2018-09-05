set(TICK_ONLINE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/online)

add_library(tick_online EXCLUDE_FROM_ALL
        online_forest_classifier.cpp
        ${TICK_ONLINE_INCLUDE_DIR}/online_forest_classifier.h
        online_forest_regressor.cpp
        ${TICK_ONLINE_INCLUDE_DIR}/online_forest_regressor.h)
