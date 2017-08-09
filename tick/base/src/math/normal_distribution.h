//
// Created by Martin Bompaire on 11/04/16.
//

#ifndef TICK_BASE_SRC_MATH_NORMAL_DISTRIBUTION_H_
#define TICK_BASE_SRC_MATH_NORMAL_DISTRIBUTION_H_

// License: BSD 3 clause

#include "array.h"

extern double standard_normal_cdf(double x);

extern DLL_PUBLIC double standard_normal_inv_cdf(const double q);

extern void standard_normal_inv_cdf(ArrayDouble &q, ArrayDouble &out);

#endif  // TICK_BASE_SRC_MATH_NORMAL_DISTRIBUTION_H_
