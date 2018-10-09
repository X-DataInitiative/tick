// License: BSD 3 clause

%module hawkes_simulation

%include tick/base/defs.i
%include tick/base/serialization.i

%{
#include "tick/base/tick_python.h"
%}

%import(module="tick.base") tick/base/base_module.i

%include simu_point_process.i
%include simu_poisson_process.i
%include simu_inhomogeneous_poisson.i
%include simu_hawkes.i
%include hawkes_kernels.i
