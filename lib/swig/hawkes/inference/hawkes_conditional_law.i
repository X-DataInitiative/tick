// License: BSD 3 clause

%{
  #include "tick/hawkes/inference/hawkes_conditional_law.h"
%}

extern void PointProcessCondLaw(ArrayDouble &y_time,
                                ArrayDouble &z_time, ArrayDouble &z_mark,
                                ArrayDouble &lags,
                                double zmin, double zmax,
                                double y_T,
                                double y_lambda,
                                ArrayDouble &res_X, ArrayDouble &res_Y);
