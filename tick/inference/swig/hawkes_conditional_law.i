%{
  #include "hawkes_conditional_law.h"
%}

extern void PointProcessCondLaw(ArrayDouble &y_time, ArrayDouble & y_mark, ArrayDouble & z_time, ArrayDouble & z_mark, ArrayDouble & lags,double zmin,double zmax, ArrayDouble & res_X, ArrayDouble & res_Y);

