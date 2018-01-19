set(TICK_BASE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/base)

add_library(tick_base EXCLUDE_FROM_ALL
        ${TICK_BASE_INCLUDE_DIR}/time_func.h
        time_func.cpp

        ${TICK_BASE_INCLUDE_DIR}/interruption.h
        interruption.cpp

        ${TICK_BASE_INCLUDE_DIR}/parallel/parallel.h
        ${TICK_BASE_INCLUDE_DIR}/parallel/parallel_utils.h

        ${TICK_BASE_INCLUDE_DIR}/exceptions_test.h
        exceptions_test.cpp

        ${TICK_BASE_INCLUDE_DIR}/base.h
        ${TICK_BASE_INCLUDE_DIR}/base_test.h

        ${TICK_BASE_INCLUDE_DIR}/debug.h
        ${TICK_BASE_INCLUDE_DIR}/defs.h

        ${TICK_BASE_INCLUDE_DIR}/tick_python.h
        ${TICK_BASE_INCLUDE_DIR}/serialization.h

        ${TICK_BASE_INCLUDE_DIR}/math/normal_distribution.h
        math/normal_distribution.cpp
        ${TICK_BASE_INCLUDE_DIR}/math/t2exp.h
        ${TICK_BASE_INCLUDE_DIR}/math/t2exp.inl
        math/t2exp.cpp
        )
