// License: BSD 3 clause

%module survival

%include base_model_module.i

%shared_ptr(ModelCoxRegPartialLik);
%shared_ptr(ModelSCCS);

%{
#include "tick/base/tick_python.h"
%}

%import(module="tick.base") base_module.i

%include model_coxreg_partial_lik.i

%include model_sccs.i
