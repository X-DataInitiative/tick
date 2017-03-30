add_library(base EXCLUDE_FROM_ALL
        time_func.h
        time_func.cpp

        interruption.h
        interruption.cpp

        parallel/parallel.h
        parallel/parallel_utils.h

        exceptions_test.h
        exceptions_test.cpp

        base.h
        base_test.h

        debug.h
        defs.h

        tick_python.h
        serialization.h

        math/normal_distribution.h
        math/normal_distribution.cpp
        math/t2exp.h
        math/t2exp.inl
        math/t2exp.cpp
        )
