//
// Created by Martin Bompaire on 11/04/16.
//

#ifndef LIB_INCLUDE_TICK_BASE_MATH_NORMAL_DISTRIBUTION_H_
#define LIB_INCLUDE_TICK_BASE_MATH_NORMAL_DISTRIBUTION_H_

// License: BSD 3 clause

#include "tick/array/array.h"

extern double standard_normal_cdf(double x);

extern DLL_PUBLIC double standard_normal_inv_cdf(const double q);

extern void standard_normal_inv_cdf(ArrayDouble &q, ArrayDouble &out);

#endif  // LIB_INCLUDE_TICK_BASE_MATH_NORMAL_DISTRIBUTION_H_
