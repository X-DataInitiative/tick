//
// Created by Martin Bompaire on 21/05/15.
//

#ifndef LIB_INCLUDE_TICK_ARRAY_TEST_VARRAYCONTAINER_H_
#define LIB_INCLUDE_TICK_ARRAY_TEST_VARRAYCONTAINER_H_

// License: BSD 3 clause

#include "tick/array/varray.h"
#include "tick/array/array.h"
#include "array_test.h"

// Test classes in order to see if a varray is correctly deallocated

class VarrayContainer {
 public:
    VArrayDoublePtr varrayPtr;
    VarrayContainer() {}
    void initVarray();
    void initVarray(int size);
    std::int64_t nRef();
};

class VarrayUser {
 public:
    VArrayDoublePtr varrayPtr;
    VarrayUser() {}
    std::int64_t nRef();
    void setArray(VarrayContainer vcc);
};

#endif  // LIB_INCLUDE_TICK_ARRAY_TEST_VARRAYCONTAINER_H_
