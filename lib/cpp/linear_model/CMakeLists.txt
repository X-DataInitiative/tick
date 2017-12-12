set(TICK_LINEAR_MODEL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/linear_model)

add_library(tick_linear_model EXCLUDE_FROM_ALL
        ${TICK_LINEAR_MODEL_INCLUDE_DIR}/model_linreg.h
        ${TICK_LINEAR_MODEL_INCLUDE_DIR}/model_logreg.h
        ${TICK_LINEAR_MODEL_INCLUDE_DIR}/model_hinge.h
        ${TICK_LINEAR_MODEL_INCLUDE_DIR}/model_quadratic_hinge.h
        ${TICK_LINEAR_MODEL_INCLUDE_DIR}/model_smoothed_hinge.h
        ${TICK_LINEAR_MODEL_INCLUDE_DIR}/model_poisreg.h
        model_hinge.cpp 
        model_quadratic_hinge.cpp 
        model_smoothed_hinge.cpp
		model_linreg.cpp
		model_logreg.cpp
		model_poisreg.cpp
)

target_link_libraries(tick_linear_model
		${TICK_LIB_BASE_MODEL}
		${TICK_LIB_ARRAY})
