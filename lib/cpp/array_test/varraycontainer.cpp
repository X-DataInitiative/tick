// License: BSD 3 clause

//
// Created by Martin Bompaire on 21/05/15.
//

#include "tick/array_test/varraycontainer.h"

void VarrayContainer::initVarray() {
    varrayPtr = test_VArrayDouble_append1(10);
}

void VarrayContainer::initVarray(int size) {
    varrayPtr = test_VArrayDouble_append1(size);
}

std::int64_t VarrayContainer::nRef() {
    if (varrayPtr)
        return varrayPtr.use_count();
    else
        return 0;
}

std::int64_t VarrayUser::nRef() {
    if (varrayPtr)
        return varrayPtr.use_count();
    else
        return 0;
}

void VarrayUser::setArray(VarrayContainer vcc) {
    varrayPtr = vcc.varrayPtr;
}
