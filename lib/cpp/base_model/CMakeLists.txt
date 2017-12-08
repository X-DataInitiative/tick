set(TICK_BASE_MODEL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/base_model)

add_library(tick_base_model EXCLUDE_FROM_ALL
        ${TICK_BASE_MODEL_INCLUDE_DIR}/model.h
        ${TICK_BASE_MODEL_INCLUDE_DIR}/model_labels_features.h
        ${TICK_BASE_MODEL_INCLUDE_DIR}/model_generalized_linear.h
        ${TICK_BASE_MODEL_INCLUDE_DIR}/model_lipschitz.h
        model_generalized_linear.cpp 
        model_labels_features.cpp 
        model_lipschitz.cpp
)


target_link_libraries(tick_base_model ${TICK_LIB_BASE})