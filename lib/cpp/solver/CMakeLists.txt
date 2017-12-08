set(TICK_SOLVER_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/solver)

add_library(tick_solver EXCLUDE_FROM_ALL
        ${TICK_SOLVER_INCLUDE_DIR}/sgd.h
        sgd.cpp
        ${TICK_SOLVER_INCLUDE_DIR}/svrg.h
        svrg.cpp
        ${TICK_SOLVER_INCLUDE_DIR}/saga.h
        saga.cpp
        ${TICK_SOLVER_INCLUDE_DIR}/sdca.h
        sdca.cpp
        ${TICK_SOLVER_INCLUDE_DIR}/adagrad.h
        adagrad.cpp
        ${TICK_SOLVER_INCLUDE_DIR}/sto_solver.h
        sto_solver.cpp)
