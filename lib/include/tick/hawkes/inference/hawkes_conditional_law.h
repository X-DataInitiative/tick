
#ifndef LIB_INCLUDE_TICK_HAWKES_INFERENCE_HAWKES_CONDITIONAL_LAW_H_
#define LIB_INCLUDE_TICK_HAWKES_INFERENCE_HAWKES_CONDITIONAL_LAW_H_

// License: BSD 3 clause

#include "tick/base/base.h"

extern void PointProcessCondLaw(ArrayDouble &y_time,
                                ArrayDouble &z_time, ArrayDouble &z_mark,
                                ArrayDouble &lags,
                                double zmin, double zmax,
                                double y_T,
                                double y_lambda,
                                ArrayDouble &res_X, ArrayDouble &res_Y);

#endif  // LIB_INCLUDE_TICK_HAWKES_INFERENCE_HAWKES_CONDITIONAL_LAW_H_
