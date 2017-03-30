//
// Created by Martin Bompaire on 21/05/15.
//

#include "varraycontainer.h"

void VarrayContainer::initVarray() {
    varrayPtr = test_VArrayDouble_append1(10);
}

void VarrayContainer::initVarray(int size) {
    varrayPtr = test_VArrayDouble_append1(size);
}

long VarrayContainer::nRef() {
    if (varrayPtr)
        return varrayPtr.use_count();
    else
        return 0;
}

long VarrayUser::nRef() {
    if (varrayPtr)
        return varrayPtr.use_count();
    else
        return 0;
}

void VarrayUser::setArray(VarrayContainer vcc){
    varrayPtr = vcc.varrayPtr;
}
