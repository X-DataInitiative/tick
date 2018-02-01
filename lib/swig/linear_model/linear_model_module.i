// License: BSD 3 clause

%module linear_model

%include base_model_module.i

%shared_ptr(TModelLinReg<double, double>);
%shared_ptr(TModelLinReg<float, float>);
%shared_ptr(ModelLinReg);

%shared_ptr(TModelLogReg<double, double>);
%shared_ptr(TModelLogReg<float, float>);
%shared_ptr(ModelLogReg);

%shared_ptr(TModelPoisReg<double, double>);
%shared_ptr(TModelPoisReg<float, float>);
%shared_ptr(ModelPoisReg);

%shared_ptr(ModelHinge);
%shared_ptr(ModelSmoothedHinge);
%shared_ptr(ModelQuadraticHinge);

%{
#include "tick/base/tick_python.h"
%}

%import(module="tick.base") base_module.i

%include model_hinge.i

%include model_quadratic_hinge.i

%include model_smoothed_hinge.i

%include model_linreg.i

%include model_logreg.i

%include model_poisreg.i
