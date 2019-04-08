// License: BSD 3 clause

%module survival

%include tick/base_model/base_model_module.i

%shared_ptr(ModelCoxRegPartialLikDouble);
%shared_ptr(ModelCoxRegPartialLikFloat);

%shared_ptr(ModelSCCS);

%{
#include "tick/base/tick_python.h"
%}

%import(module="tick.base") tick/base/base_module.i

%include model_coxreg_partial_lik.i

%include model_sccs.i
