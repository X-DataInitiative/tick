

#ifndef TICK_INFERENCE_SRC_HAWKES_CONDITIONAL_LAW_H_
#define TICK_INFERENCE_SRC_HAWKES_CONDITIONAL_LAW_H_

#include "base.h"

extern void PointProcessCondLaw(ArrayDouble &y_time, ArrayDouble &y_mark,
                                ArrayDouble &z_time, ArrayDouble &z_mark,
                                ArrayDouble &lags,
                                double zmin, double zmax,
                                ArrayDouble &res_X,  ArrayDouble &res_Y);

#endif  // TICK_INFERENCE_SRC_HAWKES_CONDITIONAL_LAW_H_
