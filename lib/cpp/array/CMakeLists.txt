set(TICK_ARRAY_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/array)

add_library(tick_array EXCLUDE_FROM_ALL
        ${TICK_ARRAY_INCLUDE_DIR}/abstractarray1d2d.h
        ${TICK_ARRAY_INCLUDE_DIR}/array.h
        ${TICK_ARRAY_INCLUDE_DIR}/array2d.h
        ${TICK_ARRAY_INCLUDE_DIR}/basearray.h
        ${TICK_ARRAY_INCLUDE_DIR}/basearray2d.h
        ${TICK_ARRAY_INCLUDE_DIR}/dot.h
        ${TICK_ARRAY_INCLUDE_DIR}/sarray.h
        ${TICK_ARRAY_INCLUDE_DIR}/sarray2d.h
        ${TICK_ARRAY_INCLUDE_DIR}/sbasearray.h
        ${TICK_ARRAY_INCLUDE_DIR}/sbasearray2d.h
        ${TICK_ARRAY_INCLUDE_DIR}/sparsearray.h
        ${TICK_ARRAY_INCLUDE_DIR}/sparsearray2d.h
        ${TICK_ARRAY_INCLUDE_DIR}/ssparsearray.h
        ${TICK_ARRAY_INCLUDE_DIR}/ssparsearray2d.h
        ${TICK_ARRAY_INCLUDE_DIR}/varray.h
        ${TICK_ARRAY_INCLUDE_DIR}/view.h
        ${TICK_ARRAY_INCLUDE_DIR}/view2d.h
        ${TICK_ARRAY_INCLUDE_DIR}/carray_python.h
        ${TICK_ARRAY_INCLUDE_DIR}/vector_operations.h
        ${TICK_ARRAY_INCLUDE_DIR}/alloc.h
        alloc.cpp
        )