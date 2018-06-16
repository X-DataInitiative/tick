// License: BSD 3 clause

%module base_model

%include defs.i
%include serialization.i
%include std_shared_ptr.i

%shared_ptr(TModel<double, double>);
%shared_ptr(TModel<float, float>);
%shared_ptr(TModel<double, std::atomic<double>>);
%shared_ptr(TModel<float, std::atomic<float>>);

%shared_ptr(TModelLabelsFeatures<double, double>);
%shared_ptr(TModelLabelsFeatures<float, float>);
%shared_ptr(TModelLabelsFeatures<double, std::atomic<double>>);
%shared_ptr(TModelLabelsFeatures<float, std::atomic<float>>);
%shared_ptr(ModelLabelsFeatures);

%shared_ptr(TModelGeneralizedLinear<double, double>);
%shared_ptr(TModelGeneralizedLinear<float, float>);
%shared_ptr(TModelGeneralizedLinear<double, std::atomic<double>>);
%shared_ptr(TModelGeneralizedLinear<float, std::atomic<float>>);
%shared_ptr(ModelGeneralizedLinear);

%shared_ptr(TModelLipschitz<double, double>);
%shared_ptr(TModelLipschitz<float, float>);
%shared_ptr(TModelLipschitz<double, std::atomic<double>>);
%shared_ptr(TModelLipschitz<float, std::atomic<float>>);
%shared_ptr(ModelLipschitz);

%{
#include "tick/base/tick_python.h"
%}

%import(module="tick.base") base_module.i

%include model.i

%include model_labels_features.i

%include model_lipschitz.i

%include model_generalized_linear.i


