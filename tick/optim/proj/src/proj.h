#include "array.h"



extern unsigned int proj_half_spaces(ArrayDouble & coeffs, ArrayDouble2d &A, ArrayDouble & b,
                             ArrayDouble & norms, ArrayDouble & out,
                             const unsigned int max_pass, ArrayDouble & history);


extern void proj_simplex(ArrayDouble & coeffs, ArrayDouble & out, double r);
