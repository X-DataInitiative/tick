add_library(tick_solver EXCLUDE_FROM_ALL
        sgd.h sgd.cpp
        svrg.h svrg.cpp
        saga.h saga.cpp
        sdca.h sdca.cpp
        adagrad.h adagrad.cpp
        sto_solver.h sto_solver.cpp)

target_link_libraries(tick_solver
        ${TICK_LIB_CRANDOM}
        ${TICK_LIB_PROX}
        ${TICK_LIB_MODEL}
        )