// License: BSD 3 clause

%module base_model

%include defs.i
%include serialization.i
%include std_shared_ptr.i

%shared_ptr(Model);
%shared_ptr(ModelLabelsFeatures);
%shared_ptr(ModelGeneralizedLinear);
%shared_ptr(ModelLipschitz);

%{
#include "tick/base/tick_python.h"
%}

%import(module="tick.base") base_module.i

%include model.i

%include model_labels_features.i

%include model_lipschitz.i

%include model_generalized_linear.i


