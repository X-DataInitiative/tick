#ifndef LIB_INCLUDE_TICK_ARRAY_ALLOC_H_
#define LIB_INCLUDE_TICK_ARRAY_ALLOC_H_

// License: BSD 3 clause

/** @file */

#include "tick/base/debug.h"

#if defined(PYTHON_LINK)

#include <Python.h>
#define PYDECREF(ref) Py_DECREF(reinterpret_cast<PyObject*>(ref))
#define PYINCREF(ref) Py_INCREF(reinterpret_cast<PyObject*>(ref))

namespace tick {

template<typename T>
void python_free(T *&ptr) {
  PyMem_RawFree(reinterpret_cast<void*>(ptr));

  ptr = nullptr;
}

template<typename T>
void python_malloc(T *&ptr, ulong n) {
  ptr = (n == 0) ? nullptr : reinterpret_cast<T*>(PyMem_RawMalloc(n * sizeof(T)));
}

}  // namespace tick

#else
#define PYDECREF(ref)
#define PYINCREF(ref)

namespace tick {

template<typename T>
void python_free(T *&ptr) {
  delete[](ptr);

  ptr = nullptr;
}

template<typename T>
void python_malloc(T *&ptr, ulong n) {
  ptr = (n == 0) ? nullptr : new T[n];
}

}  // namespace tick

#endif

#if defined(DEBUG_C_ARRAY)

class DEBUGCAllocArrayCount_t {
    std::int64_t count;
 public:
    DEBUGCAllocArrayCount_t() : count(0) {}

    void add() { count++; }
    void remove() { count--; }

    std::int64_t get() const { return count; }
};

extern DEBUGCAllocArrayCount_t DEBUGCAllocArrayCount;

#define TICK_PYTHON_FREE(ptr) {\
    (tick::python_free(ptr)); \
    DEBUGCAllocArrayCount.remove(); \
    TICK_WARNING() << "C-array Free ptr=" << ptr << " --> #AllocArrayCount=" << DEBUGCAllocArrayCount.get(); \
}

#define TICK_PYTHON_MALLOC(ptr, type, n) {\
    (tick::python_malloc<type>(ptr, n)); \
    DEBUGCAllocArrayCount.add(); \
    TICK_WARNING() << "C-array Alloc size=" << n << " ptr=" << ptr << " --> #AllocArrayCount=" << DEBUGCAllocArrayCount.get(); \
}

#else

#define TICK_PYTHON_FREE(ptr)             (tick::python_free(ptr))
#define TICK_PYTHON_MALLOC(ptr, type, n)  (tick::python_malloc<type>(ptr, n))

#endif  // DEBUG_C_ARRAY

#endif  // LIB_INCLUDE_TICK_ARRAY_ALLOC_H_
