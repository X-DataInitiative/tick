// License: BSD 3 clause

%module linear_model

%include base_model_module.i

%shared_ptr(ModelLinReg);
%shared_ptr(TModelLinReg<double, double>);
%shared_ptr(TModelLinReg<float, float>);
%shared_ptr(ModelLinRegDouble);
%shared_ptr(ModelLinRegFloat);
%shared_ptr(TModelLinReg<float, std::atomic<float>>);
%shared_ptr(TModelLinReg<double, std::atomic<double>>);
%shared_ptr(ModelLinRegAtomicDouble);
%shared_ptr(ModelLinRegAtomicFloat);


%shared_ptr(ModelLogReg);
%shared_ptr(TModelLogReg<double, double>);
%shared_ptr(TModelLogReg<float, float>);
%shared_ptr(ModelLogRegDouble);
%shared_ptr(ModelLogRegFloat);
%shared_ptr(TModelLogReg<float, std::atomic<float>>);
%shared_ptr(TModelLogReg<double, std::atomic<double>>);
%shared_ptr(ModelLogRegAtomicDouble);
%shared_ptr(ModelLogRegAtomicFloat);

%shared_ptr(ModelPoisReg);
%shared_ptr(TModelPoisReg<double, double>);
%shared_ptr(TModelPoisReg<float, float>);
%shared_ptr(ModelPoisRegDouble);
%shared_ptr(ModelPoisRegFloat);

%shared_ptr(ModelHinge);
%shared_ptr(TModelHinge<double, double>);
%shared_ptr(TModelHinge<float, float>);
%shared_ptr(ModelHingeDouble);
%shared_ptr(ModelHingeFloat);

%shared_ptr(ModelSmoothedHinge);
%shared_ptr(TModelSmoothedHinge<double, double>);
%shared_ptr(TModelSmoothedHinge<float, float>);
%shared_ptr(ModelSmoothedHingeDouble);
%shared_ptr(ModelSmoothedHingeFloat);

%shared_ptr(ModelQuadraticHinge);
%shared_ptr(TModelQuadraticHinge<double, double>);
%shared_ptr(TModelQuadraticHinge<float, float>);
%shared_ptr(ModelQuadraticHingeDouble);
%shared_ptr(ModelQuadraticHingeFloat);

%{
#include "tick/base/tick_python.h"
#include "tick/base/serialization.h"
%}

%import(module="tick.base") base_module.i

%include serialization.i

%include model_hinge.i

%include model_quadratic_hinge.i

%include model_smoothed_hinge.i

%include model_linreg.i

%include model_logreg.i

%include model_poisreg.i
