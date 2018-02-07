// License: BSD 3 clause

%module robust

%include linear_model_module.i

%shared_ptr(TModelGeneralizedLinearWithIntercepts<double>);
%shared_ptr(TModelGeneralizedLinearWithIntercepts<float>);
%shared_ptr(ModelGeneralizedLinearWithInterceptsDouble);
%shared_ptr(ModelGeneralizedLinearWithInterceptsFloat);
%shared_ptr(ModelGeneralizedLinearWithIntercepts);

%shared_ptr(TModelLinRegWithIntercepts<double>);
%shared_ptr(TModelLinRegWithIntercepts<float>);
%shared_ptr(ModelLinRegWithInterceptsDouble);
%shared_ptr(ModelLinRegWithInterceptsFloat);
%shared_ptr(ModelLinRegWithIntercepts);

%shared_ptr(TModelHuber<double>);
%shared_ptr(TModelHuber<float>);
%shared_ptr(ModelHuberDouble);
%shared_ptr(ModelHuberFloat);
%shared_ptr(ModelHuber);

%shared_ptr(TModelModifiedHuber<double>);
%shared_ptr(TModelModifiedHuber<float>);
%shared_ptr(ModelModifiedHuberDouble);
%shared_ptr(ModelModifiedHuberFloat);
%shared_ptr(ModelModifiedHuber);

%shared_ptr(TModelEpsilonInsensitive<double>);
%shared_ptr(TModelEpsilonInsensitive<float>);
%shared_ptr(ModelEpsilonInsensitiveDouble);
%shared_ptr(ModelEpsilonInsensitiveFloat);
%shared_ptr(ModelEpsilonInsensitive);

%shared_ptr(TModelAbsoluteRegression<double>);
%shared_ptr(TModelAbsoluteRegression<float>);
%shared_ptr(ModelAbsoluteRegressionDouble);
%shared_ptr(ModelAbsoluteRegressionFloat);
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
