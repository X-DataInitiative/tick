// License: BSD 3 clause

%module array_test

%include defs.i

%{
#include "tick_python.h"
#include "varraycontainer.h"
#include "array_test.h"
#include "typemap_test.h"
#include "sbasearray_container.h"
%}

%import(module="tick.base.array.build.array") array_module.i

%include "array_test.h"
%include "typemap_test.h"

class VarrayUser {
public:
    VArrayDoublePtr varrayPtr;
    VarrayUser() {};
    long nRef();
    void setArray(VarrayContainer vcc);
};

class VarrayContainer {
public:
    VArrayDoublePtr varrayPtr;
    VarrayContainer() {};
    long nRef();
    void initVarray();
    void initVarray(int size);
};

%include performance_test.i

extern void test_sbasearray_container_new(SBaseArrayDoublePtr a);
extern void test_sbasearray_container_clear();
extern double test_sbasearray_container_compute();

extern void test_sbasearray2d_container_new(SBaseArrayDouble2dPtr a);
extern void test_sbasearray2d_container_clear();
extern double test_sbasearray2d_container_compute();
