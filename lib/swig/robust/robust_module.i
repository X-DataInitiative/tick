// License: BSD 3 clause

%module robust

%include linear_model_module.i

%shared_ptr(ModelGeneralizedLinearWithIntercepts);
%shared_ptr(ModelLinRegWithIntercepts);

%shared_ptr(ModelHuber);
%shared_ptr(ModelModifiedHuber);
%shared_ptr(ModelEpsilonInsensitive);
%shared_ptr(ModelAbsoluteRegression);


%{
#include "tick/base/tick_python.h"
%}

%import(module="tick.base") base_module.i

%include model_epsilon_insensitive.i

%include model_huber.i

%include model_modified_huber.i

%include model_absolute_regression.i

%include model_generalized_linear_with_intercepts.i

%include model_linreg_with_intercepts.i
