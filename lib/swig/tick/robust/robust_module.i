// License: BSD 3 clause

%module robust

%include tick/linear_model/linear_model_module.i

%shared_ptr(TModelGeneralizedLinearWithIntercepts<double, double>);
%shared_ptr(TModelGeneralizedLinearWithIntercepts<float, float>);
%shared_ptr(ModelGeneralizedLinearWithInterceptsDouble);
%shared_ptr(ModelGeneralizedLinearWithInterceptsFloat);
%shared_ptr(ModelGeneralizedLinearWithIntercepts);

%shared_ptr(TModelLinRegWithIntercepts<double, double>);
%shared_ptr(TModelLinRegWithIntercepts<float, float>);
%shared_ptr(ModelLinRegWithInterceptsDouble);
%shared_ptr(ModelLinRegWithInterceptsFloat);
%shared_ptr(ModelLinRegWithIntercepts);

%shared_ptr(TModelHuber<double, double>);
%shared_ptr(TModelHuber<float, float>);
%shared_ptr(ModelHuberDouble);
%shared_ptr(ModelHuberFloat);
%shared_ptr(ModelHuber);

%shared_ptr(TModelModifiedHuber<double, double>);
%shared_ptr(TModelModifiedHuber<float, float>);
%shared_ptr(ModelModifiedHuberDouble);
%shared_ptr(ModelModifiedHuberFloat);
%shared_ptr(ModelModifiedHuber);

%shared_ptr(TModelEpsilonInsensitive<double, double>);
%shared_ptr(TModelEpsilonInsensitive<float, float>);
%shared_ptr(ModelEpsilonInsensitiveDouble);
%shared_ptr(ModelEpsilonInsensitiveFloat);
%shared_ptr(ModelEpsilonInsensitive);

%shared_ptr(TModelAbsoluteRegression<double, double>);
%shared_ptr(TModelAbsoluteRegression<float, float>);
%shared_ptr(ModelAbsoluteRegressionDouble);
%shared_ptr(ModelAbsoluteRegressionFloat);
%shared_ptr(ModelAbsoluteRegression);


%{
#include "tick/base/tick_python.h"
%}

%import(module="tick.base") tick/base/base_module.i

%include model_epsilon_insensitive.i

%include model_huber.i

%include model_modified_huber.i

%include model_absolute_regression.i

%include model_generalized_linear_with_intercepts.i

%include model_linreg_with_intercepts.i
