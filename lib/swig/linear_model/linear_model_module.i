// License: BSD 3 clause

%module linear_model

%include base_model_module.i

%shared_ptr(ModelLinReg);
%shared_ptr(TModelLinReg<double>);
%shared_ptr(TModelLinReg<float>);
%shared_ptr(ModelLinRegDouble);
%shared_ptr(ModelLinRegFloat);

%shared_ptr(ModelLogReg);
%shared_ptr(TModelLogReg<double>);
%shared_ptr(TModelLogReg<float>);
%shared_ptr(ModelLogRegDouble);
%shared_ptr(ModelLogRegFloat);

%shared_ptr(ModelPoisReg);
%shared_ptr(TModelPoisReg<double>);
%shared_ptr(TModelPoisReg<float>);
%shared_ptr(ModelPoisRegDouble);
%shared_ptr(ModelPoisRegFloat);

%shared_ptr(ModelHinge);
%shared_ptr(TModelHinge<double>);
%shared_ptr(TModelHinge<float>);
%shared_ptr(ModelHingeDouble);
%shared_ptr(ModelHingeFloat);

%shared_ptr(ModelSmoothedHinge);
%shared_ptr(TModelSmoothedHinge<double>);
%shared_ptr(TModelSmoothedHinge<float>);
%shared_ptr(ModelSmoothedHingeDouble);
%shared_ptr(ModelSmoothedHingeFloat);

%shared_ptr(ModelQuadraticHinge);
%shared_ptr(TModelQuadraticHinge<double>);
%shared_ptr(TModelQuadraticHinge<float>);
%shared_ptr(ModelQuadraticHingeDouble);
%shared_ptr(ModelQuadraticHingeFloat);

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
