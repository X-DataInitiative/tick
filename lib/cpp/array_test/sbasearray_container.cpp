// License: BSD 3 clause

//
//  ssparsearray_container.cpp
//
//  Created by bacry on 11/01/2016.
//  Copyright (c) 2016 bacry. All rights reserved.
//

#include "tick/array_test/sbasearray_container.h"

// Test classes in order to see if a ssparsearray is correctly deallocated

class SAbstractArrayContainer {
 public:
    SBaseArrayDoublePtr arrayPtr;

    SAbstractArrayContainer() {}

    void init(SBaseArrayDoublePtr arrayPtr1) {
        arrayPtr = arrayPtr1;
    }

    void clear() {
        arrayPtr = nullptr;
    }
};

static SAbstractArrayContainer container;

void test_sbasearray_container_new(SBaseArrayDoublePtr a) {
    container.init(a);
}

void test_sbasearray_container_clear() {
    container.clear();
}

double test_sbasearray_container_compute() {
    if (!container.arrayPtr) return -1;
    if (container.arrayPtr->is_dense()) return container.arrayPtr->sum();
    double res = container.arrayPtr->sum();
    SSparseArrayDoublePtr ptr = std::static_pointer_cast<SSparseArrayDouble>(container.arrayPtr);
    for (ulong i = 0; i < ptr->size_sparse(); i++) res += ptr->indices()[i];
    return res;
}


class SAbstractArray2dContainer {
 public:
    SBaseArrayDouble2dPtr arrayPtr;

    SAbstractArray2dContainer() {}

    void init(SBaseArrayDouble2dPtr arrayPtr1) {
        arrayPtr = arrayPtr1;
    }

    void clear() {
        arrayPtr = nullptr;
    }
};

static SAbstractArray2dContainer container2;

void test_sbasearray2d_container_new(SBaseArrayDouble2dPtr a) {
    container2.init(a);
}

void test_sbasearray2d_container_clear() {
    container2.clear();
}

double test_sbasearray2d_container_compute() {
    if (!container2.arrayPtr) return -1;
    if (container2.arrayPtr->is_dense()) return container2.arrayPtr->sum();
    double res = container2.arrayPtr->sum();
    SSparseArrayDouble2dPtr ptr = std::static_pointer_cast<SSparseArrayDouble2d>(container2.arrayPtr);
    for (ulong i = 0; i < ptr->size_sparse(); i++) res += ptr->indices()[i];
    for (ulong i = 0; i <= ptr->n_rows(); i++) res += ptr->row_indices()[i];
    return res;
}
