set(TICK_SIMULATION_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/simulation)

add_library(tick_simulation EXCLUDE_FROM_ALL
        pp.cpp 
        ${TICK_SIMULATION_INCLUDE_DIR}/pp.h
        poisson.cpp 
        ${TICK_SIMULATION_INCLUDE_DIR}/poisson.h
        hawkes.cpp 
        ${TICK_SIMULATION_INCLUDE_DIR}/hawkes.h
        inhomogeneous_poisson.cpp 
        ${TICK_SIMULATION_INCLUDE_DIR}/inhomogeneous_poisson.h
        hawkes_kernels/hawkes_kernel.cpp
        ${TICK_SIMULATION_INCLUDE_DIR}/hawkes_kernels/hawkes_kernel.h 
        hawkes_kernels/hawkes_kernel_exp.cpp
        ${TICK_SIMULATION_INCLUDE_DIR}/hawkes_kernels/hawkes_kernel_exp.h 
        hawkes_kernels/hawkes_kernel_sum_exp.cpp
        ${TICK_SIMULATION_INCLUDE_DIR}/hawkes_kernels/hawkes_kernel_sum_exp.h 
        hawkes_kernels/hawkes_kernel_power_law.cpp
        ${TICK_SIMULATION_INCLUDE_DIR}/hawkes_kernels/hawkes_kernel_power_law.h 
        ${TICK_SIMULATION_INCLUDE_DIR}/hawkes_kernels/hawkes_kernel_0.h
        hawkes_kernels/hawkes_kernel_time_func.cpp 
        ${TICK_SIMULATION_INCLUDE_DIR}/hawkes_kernels/hawkes_kernel_time_func.h
        ${TICK_SIMULATION_INCLUDE_DIR}/hawkes_baselines/baseline.h
        hawkes_baselines/constant_baseline.cpp 
        ${TICK_SIMULATION_INCLUDE_DIR}/hawkes_baselines/constant_baseline.h
        hawkes_baselines/timefunction_baseline.cpp 
        ${TICK_SIMULATION_INCLUDE_DIR}/hawkes_baselines/timefunction_baseline.h)

target_link_libraries(tick_simulation
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_BASE}
        ${TICK_LIB_CRANDOM})
