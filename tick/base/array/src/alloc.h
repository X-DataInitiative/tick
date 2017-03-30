#ifndef TICK_BASE_ARRAY_SRC_ALLOC_H_
#define TICK_BASE_ARRAY_SRC_ALLOC_H_

/** @file */

#include "debug.h"

//
// Macros PYDECREF, PYINCREF, _PYSHARED_FREE_ARRAY and _PYSHARED_ALLOC_ARRAY
//
#ifdef PYTHON_LINK
#include <Python.h>
#define PYDECREF(ref) Py_DECREF(reinterpret_cast<PyObject*>(ref))
#define PYINCREF(ref) Py_INCREF(reinterpret_cast<PyObject*>(ref))
#define _PYSHARED_FREE_ARRAY(ptr) PyMem_Free(reinterpret_cast<void*> (ptr))
#define _PYSHARED_ALLOC_ARRAY(ptr, type, n) ptr = reinterpret_cast<type*>(PyMem_Malloc((n)*sizeof(type)))
#else
#define PYDECREF(ref)
#define PYINCREF(ref)
#define _PYSHARED_FREE_ARRAY(ptr) delete[](ptr)
#define _PYSHARED_ALLOC_ARRAY(ptr, type, n) ptr = new type[n]
#endif

//
// Macros PYSHARED_FREE_ARRAY and PYSHARED_ALLOC_ARRAY
//
#ifdef DEBUG_C_ARRAY

class DEBUGCAllocArrayCount{
    static std::int64_t count;
 public:
    DEBUGCAllocArrayCount() {}
    static void add() {DEBUGCAllocArrayCount::count++;}
    static void remove() {DEBUGCAllocArrayCount::count--;}
    staticstd::int64_t get() {return DEBUGCAllocArrayCount::count;}
};

#define PYSHARED_FREE_ARRAY(ptr) {\
    if (ptr) { \
        _PYSHARED_FREE_ARRAY(ptr); \
        DEBUGCAllocArrayCount::remove(); \
        std::cout << "C-array Free ptr=" << ptr << " --> #AllocArrayCount=" \
            << DEBUGCAllocArrayCount::get() << std::endl; \
    }\
}

#define PYSHARED_ALLOC_ARRAY(ptr, type, n) {\
    if (n != 0) { \
        _PYSHARED_ALLOC_ARRAY(ptr, type, n); \
        DEBUGCAllocArrayCount::add(); \
        std::cout << "C-array Alloc size=" << n << " ptr=" << ptr << " --> #AllocArrayCount=" \
            << DEBUGCAllocArrayCount::get() << std::endl; \
    } else {ptr = nullptr;} \
}

#else

#define PYSHARED_FREE_ARRAY(ptr) _PYSHARED_FREE_ARRAY(ptr)
#define PYSHARED_ALLOC_ARRAY(ptr, type, n) \
    if (n != 0) {\
        _PYSHARED_ALLOC_ARRAY(ptr, type, n);\
    } else {\
        ptr = nullptr;\
    }

#endif

#endif  // TICK_BASE_ARRAY_SRC_ALLOC_H_
