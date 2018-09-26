// License: BSD 3 clause

%module solver

%include defs.i
%include std_shared_ptr.i

%{
#include "tick/base/tick_python.h"
%}
%import(module="tick.base") base_module.i

%{
#include "tick/base/serialization.h"
#include "tick/base_model/model.h"
#include "tick/solver/enums.h"
%}

%include serialization.i

%include sto_solver.i
%include adagrad.i

%include sdca.i
%include asdca.i
%include sgd.i

%include saga.i
%include asaga.i
%include svrg.i
