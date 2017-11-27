set(TICK_PREPROCESSING_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/preprocessing)

add_library(tick_preprocessing EXCLUDE_FROM_ALL
        sparse_longitudinal_features_product.cpp 
        ${TICK_PREPROCESSING_INCLUDE_DIR}/sparse_longitudinal_features_product.h
        longitudinal_features_lagger.cpp
        ${TICK_PREPROCESSING_INCLUDE_DIR}/longitudinal_features_lagger.h)
