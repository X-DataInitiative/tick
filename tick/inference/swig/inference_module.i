// License: BSD 3 clause

%module inference

%include defs.i
%include serialization.i

%{
  #include "tick_python.h"
%}

%import(module="tick.base") base_module.i

%{
#include "base/hawkes_list.h"
%}

// Is there a cleaner way to make our learners inherit from ModelHawkesList ?
%include model_module.i

%include hawkes_conditional_law.i
%include hawkes_em.i
%include hawkes_adm4.i
%include hawkes_basis_kernels.i
%include hawkes_sumgaussians.i
%include hawkes_sdca_loglik.i