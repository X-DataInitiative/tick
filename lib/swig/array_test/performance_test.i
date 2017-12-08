// License: BSD 3 clause

%{
#include "tick/array_test/performance_test.h"
%}

extern double test_sum_double_pointer(unsigned long size, unsigned long n_loops);
extern double test_sum_ArrayDouble(unsigned long size, unsigned long n_loops);
extern double test_sum_SArray_shared_ptr(unsigned long size, unsigned long n_loops);
extern double test_sum_VArray_shared_ptr(unsigned long size, unsigned long n_loops);