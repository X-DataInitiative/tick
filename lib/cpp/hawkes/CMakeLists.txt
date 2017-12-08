set(TICK_HAWKES_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/hawkes)

add_library(tick_hawkes EXCLUDE_FROM_ALL
    ${TICK_HAWKES_INCLUDE_DIR}/inference/hawkes_conditional_law.h
    ${TICK_HAWKES_INCLUDE_DIR}/inference/hawkes_em.h
    ${TICK_HAWKES_INCLUDE_DIR}/inference/hawkes_adm4.h
    ${TICK_HAWKES_INCLUDE_DIR}/inference/hawkes_basis_kernels.h
    ${TICK_HAWKES_INCLUDE_DIR}/inference/hawkes_sumgaussians.h
    inference/hawkes_adm4.cpp
    inference/hawkes_basis_kernels.cpp 
    inference/hawkes_conditional_law.cpp
    inference/hawkes_em.cpp 
    inference/hawkes_sumgaussians.cpp


    ${TICK_HAWKES_INCLUDE_DIR}/model/base/hawkes_model.h
    ${TICK_HAWKES_INCLUDE_DIR}/model/base/hawkes_fixed_kern_loglik.h
    ${TICK_HAWKES_INCLUDE_DIR}/model/hawkes_fixed_expkern_loglik.h 
    ${TICK_HAWKES_INCLUDE_DIR}/model/hawkes_fixed_sumexpkern_loglik.h 
    ${TICK_HAWKES_INCLUDE_DIR}/model/hawkes_fixed_expkern_leastsq.h 
    ${TICK_HAWKES_INCLUDE_DIR}/model/hawkes_fixed_sumexpkern_leastsq.h 
    ${TICK_HAWKES_INCLUDE_DIR}/model/hawkes_utils.h 
    ${TICK_HAWKES_INCLUDE_DIR}/model/base/hawkes_list.h 
    ${TICK_HAWKES_INCLUDE_DIR}/model/variants/hawkes_leastsq_list.h 
    ${TICK_HAWKES_INCLUDE_DIR}/model/variants/hawkes_fixed_expkern_leastsq_list.h 
    ${TICK_HAWKES_INCLUDE_DIR}/model/variants/hawkes_fixed_expkern_loglik_list.h 
    ${TICK_HAWKES_INCLUDE_DIR}/model/variants/hawkes_fixed_kern_loglik_list.h 
    ${TICK_HAWKES_INCLUDE_DIR}/model/base/hawkes_single.h
    ${TICK_HAWKES_INCLUDE_DIR}/model/variants/hawkes_fixed_sumexpkern_leastsq_list.h 
    ${TICK_HAWKES_INCLUDE_DIR}/model/variants/hawkes_fixed_sumexpkern_loglik_list.h 
    model/variants/hawkes_fixed_sumexpkern_loglik_list.cpp
    model/variants/hawkes_fixed_sumexpkern_leastsq_list.cpp
    model/variants/hawkes_fixed_kern_loglik_list.cpp
    model/variants/hawkes_fixed_expkern_loglik_list.cpp
    model/variants/hawkes_leastsq_list.cpp
    model/variants/hawkes_fixed_expkern_leastsq_list.cpp
    model/base/hawkes_fixed_kern_loglik.cpp 
    model/base/hawkes_single.cpp 
    model/base/hawkes_model.cpp 
    model/base/hawkes_list.cpp
    model/hawkes_fixed_sumexpkern_leastsq.cpp
    model/hawkes_fixed_expkern_leastsq.cpp
    model/hawkes_fixed_sumexpkern_loglik.cpp
    model/hawkes_fixed_expkern_loglik.cpp
    model/hawkes_utils.cpp


    ${TICK_HAWKES_INCLUDE_DIR}/simulation/pp.h
    ${TICK_HAWKES_INCLUDE_DIR}/simulation/poisson.h
    ${TICK_HAWKES_INCLUDE_DIR}/simulation/hawkes.h
    ${TICK_HAWKES_INCLUDE_DIR}/simulation/inhomogeneous_poisson.h
    ${TICK_HAWKES_INCLUDE_DIR}/simulation/hawkes_kernels/hawkes_kernel.h 
    ${TICK_HAWKES_INCLUDE_DIR}/simulation/hawkes_kernels/hawkes_kernel_exp.h 
    ${TICK_HAWKES_INCLUDE_DIR}/simulation/hawkes_kernels/hawkes_kernel_sum_exp.h 
    ${TICK_HAWKES_INCLUDE_DIR}/simulation/hawkes_kernels/hawkes_kernel_power_law.h 
    ${TICK_HAWKES_INCLUDE_DIR}/simulation/hawkes_kernels/hawkes_kernel_0.h
    ${TICK_HAWKES_INCLUDE_DIR}/simulation/hawkes_kernels/hawkes_kernel_time_func.h
    ${TICK_HAWKES_INCLUDE_DIR}/simulation/hawkes_baselines/baseline.h
    ${TICK_HAWKES_INCLUDE_DIR}/simulation/hawkes_baselines/constant_baseline.h
    ${TICK_HAWKES_INCLUDE_DIR}/simulation/hawkes_baselines/timefunction_baseline.h
    simulation/hawkes.cpp 
    simulation/poisson.cpp 
    simulation/inhomogeneous_poisson.cpp 
    simulation/hawkes_baselines/timefunction_baseline.cpp 
    simulation/hawkes_kernels/hawkes_kernel_power_law.cpp
    simulation/hawkes_kernels/hawkes_kernel_exp.cpp
    simulation/hawkes_kernels/hawkes_kernel_time_func.cpp 
    simulation/hawkes_kernels/hawkes_kernel_sum_exp.cpp
    simulation/hawkes_kernels/hawkes_kernel.cpp
    simulation/hawkes_baselines/constant_baseline.cpp 
    simulation/pp.cpp 
)

target_link_libraries(tick_hawkes
        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_LINEAR_MODEL}
        ${TICK_LIB_CRANDOM})