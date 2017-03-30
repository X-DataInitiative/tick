add_library(simulation EXCLUDE_FROM_ALL
        pp.cpp pp.h
        poisson.cpp poisson.h
        hawkes.cpp hawkes.h
        inhomogeneous_poisson.cpp inhomogeneous_poisson.h
        hawkes_kernels/hawkes_kernel.cpp
        hawkes_kernels/hawkes_kernel.h hawkes_kernels/hawkes_kernel_exp.cpp
        hawkes_kernels/hawkes_kernel_exp.h hawkes_kernels/hawkes_kernel_sum_exp.cpp
        hawkes_kernels/hawkes_kernel_sum_exp.h hawkes_kernels/hawkes_kernel_power_law.cpp
        hawkes_kernels/hawkes_kernel_power_law.h hawkes_kernels/hawkes_kernel_0.h
        hawkes_kernels/hawkes_kernel_time_func.cpp hawkes_kernels/hawkes_kernel_time_func.h)

target_link_libraries(simulation base)
