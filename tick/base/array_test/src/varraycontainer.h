//
// Created by Martin Bompaire on 21/05/15.
//

#ifndef TICK_BASE_ARRAY_TEST_SRC_VARRAYCONTAINER_H_
#define TICK_BASE_ARRAY_TEST_SRC_VARRAYCONTAINER_H_

// License: BSD 3 clause

#include "varray.h"
#include "array.h"
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

#endif  // TICK_BASE_ARRAY_TEST_SRC_VARRAYCONTAINER_H_
