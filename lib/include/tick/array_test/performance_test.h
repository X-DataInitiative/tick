//
// Created by Martin Bompaire on 22/05/15.
//

#ifndef LIB_INCLUDE_TICK_ARRAY_TEST_PERFORMANCE_TEST_H_
#define LIB_INCLUDE_TICK_ARRAY_TEST_PERFORMANCE_TEST_H_

// License: BSD 3 clause

#include "timer.h"
#include "tick/base/base.h"


extern void test_element_access();

extern void test_element_access_inherited_array();

/// @brief compute the sum of a double* array full of one, n_loops times
extern double test_sum_double_pointer(ulong size,
                                      ulong n_loops);

/// @brief compute the sum of a ArrayDouble full of one, n_loops times
extern double test_sum_ArrayDouble(ulong size,
                                   ulong n_loops);

/// @brief compute the sum of a SArrayDoublePtr full of one, n_loops times
extern double test_sum_SArray_shared_ptr(ulong size,
                                         ulong n_loops);

/// @brief compute the sum of a VArrayDoublePtr full of one, n_loops times
extern double test_sum_VArray_shared_ptr(ulong size,
                                         ulong n_loops);

class ToyAbstractArray {
 public :
    double *val;
    ulong size;

    ToyAbstractArray() : val(0), size(0L) {}

    virtual ~ToyAbstractArray() {}

    virtual double getValue(ulong i) = 0;
};


class InheritedArray : public ToyAbstractArray {
 public:
    InheritedArray(double *val1, ulong size1) {
        val = val1;
        size = size1;
    }

    inline double &operator[](ulong i) {
        return val[i];
    }

    double getValue(ulong i) {
        return val[i];
    }
};

#endif  // LIB_INCLUDE_TICK_ARRAY_TEST_PERFORMANCE_TEST_H_
