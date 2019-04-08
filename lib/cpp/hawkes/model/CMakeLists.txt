set(TICK_HAWKES_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/hawkes/model)

add_library(
        tick_hawkes_model EXCLUDE_FROM_ALL

        ${TICK_HAWKES_INCLUDE_DIR}/base/model_hawkes.h
        ${TICK_HAWKES_INCLUDE_DIR}/base/model_hawkes_single.h
        ${TICK_HAWKES_INCLUDE_DIR}/base/model_hawkes_list.h
        ${TICK_HAWKES_INCLUDE_DIR}/base/model_hawkes_loglik_single.h
        ${TICK_HAWKES_INCLUDE_DIR}/base/model_hawkes_loglik.h
        ${TICK_HAWKES_INCLUDE_DIR}/base/model_hawkes_leastsq.h

        ${TICK_HAWKES_INCLUDE_DIR}/list_of_realizations/model_hawkes_expkern_leastsq.h
        ${TICK_HAWKES_INCLUDE_DIR}/list_of_realizations/model_hawkes_expkern_loglik.h
        ${TICK_HAWKES_INCLUDE_DIR}/list_of_realizations/model_hawkes_sumexpkern_leastsq.h
        ${TICK_HAWKES_INCLUDE_DIR}/list_of_realizations/model_hawkes_sumexpkern_loglik.h

        ${TICK_HAWKES_INCLUDE_DIR}/model_hawkes_expkern_leastsq_single.h
        ${TICK_HAWKES_INCLUDE_DIR}/model_hawkes_expkern_loglik_single.h
        ${TICK_HAWKES_INCLUDE_DIR}/model_hawkes_sumexpkern_leastsq_single.h
        ${TICK_HAWKES_INCLUDE_DIR}/model_hawkes_sumexpkern_loglik_single.h

        ${TICK_HAWKES_INCLUDE_DIR}/model_hawkes_utils.h

        base/model_hawkes_loglik_single.cpp
        base/model_hawkes_single.cpp
        base/model_hawkes.cpp
        base/model_hawkes_list.cpp
        base/model_hawkes_loglik.cpp
        base/model_hawkes_leastsq.cpp

        list_of_realizations/model_hawkes_sumexpkern_loglik.cpp
        list_of_realizations/model_hawkes_sumexpkern_leastsq.cpp
        list_of_realizations/model_hawkes_expkern_loglik.cpp
        list_of_realizations/model_hawkes_expkern_leastsq.cpp

        model_hawkes_expkern_leastsq_single.cpp
        model_hawkes_expkern_loglik_single.cpp
        model_hawkes_sumexpkern_leastsq_single.cpp
        model_hawkes_sumexpkern_loglik_single.cpp

        model_hawkes_utils.cpp
)

target_link_libraries(
        tick_hawkes_model
        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_LINEAR_MODEL}
        ${TICK_LIB_CRANDOM}
)