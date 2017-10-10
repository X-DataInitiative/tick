// License: BSD 3 clause

%module solver

%include defs.i
%include std_shared_ptr.i

%{
#include "tick_python.h"
%}
%import(module="tick.base") base_module.i

%{
#include "model.h"
%}

%include sto_solver.i
%include sgd.i
%include svrg.i
%include saga.i
%include sdca.i
%include adagrad.i
