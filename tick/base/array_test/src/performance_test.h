//
// Created by Martin Bompaire on 22/05/15.
//

#ifndef TICK_PERFORMANCE_TEST_H
#define TICK_PERFORMANCE_TEST_H

#include "timer.h"
#include "base.h"


extern void test_element_access();

extern void test_element_access_inherited_array();

///@brief compute the sum of a double* array full of one, n_loops times
extern double test_sum_double_pointer(unsigned long size,
                                      unsigned long n_loops);

///@brief compute the sum of a ArrayDouble full of one, n_loops times
extern double test_sum_ArrayDouble(unsigned long size,
                                   unsigned long n_loops);

///@brief compute the sum of a SArrayDoublePtr full of one, n_loops times
extern double test_sum_SArray_shared_ptr(unsigned long size,
                                         unsigned long n_loops);

///@brief compute the sum of a VArrayDoublePtr full of one, n_loops times
extern double test_sum_VArray_shared_ptr(unsigned long size,
                                         unsigned long n_loops);

class ToyAbstractArray {

public :
    double *val;
    unsigned long size;

    ToyAbstractArray() : val(0), size(0L) { };

    virtual ~ToyAbstractArray() { };

    virtual double getValue(unsigned long i) = 0;
};


class InheritedArray : public ToyAbstractArray {

public:
    InheritedArray(double *val1, unsigned long size1) {
        val = val1; size = size1;
    };

    inline double &operator[](unsigned long i) {
        return val[i];
    }

    double getValue(unsigned long i) {
        return val[i];
    }
};

#endif //TICK_PERFORMANCE_TEST_H
