//
// Created by Martin Bompaire on 21/05/15.
//

#ifndef TICK_VARRAYCONTAINER_H
#define TICK_VARRAYCONTAINER_H

#include "varray.h"
#include "array.h"
#include "array_test.h"

// Test classes in order to see if a varray is correctly deallocated

class VarrayContainer {
public:
    VArrayDoublePtr varrayPtr;
    VarrayContainer() {};
    void initVarray();
    void initVarray(int size);
    long nRef();
};

class VarrayUser {
public:
    VArrayDoublePtr varrayPtr;
    VarrayUser() {};
    long nRef();
    void setArray(VarrayContainer vcc);
};


#endif //TICK_VARRAYCONTAINER_H
