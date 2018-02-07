// License: BSD 3 clause

%{
#include "tick/robust/model_absolute_regression.h"
%}

template <class T>
class TModelAbsoluteRegression : public virtual TModelGeneralizedLinear<T>{
 public:
  TModelGeneralizedLinearWithIntercepts(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};

%rename(ModelAbsoluteRegression) TModelAbsoluteRegression<double>;
class ModelAbsoluteRegression : public virtual TModelGeneralizedLinear<double>{
 public:
  ModelAbsoluteRegression(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelAbsoluteRegression<double> ModelAbsoluteRegression;

%rename(ModelAbsoluteRegressionDouble) TModelAbsoluteRegression<double>;
class ModelAbsoluteRegressionDouble : public virtual TModelGeneralizedLinear<double>{
 public:
  ModelAbsoluteRegressionDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelAbsoluteRegression<double> ModelAbsoluteRegressionDouble;

%rename(ModelAbsoluteRegressionFloat) TModelAbsoluteRegression<float>;
class ModelAbsoluteRegressionFloat : public virtual TModelGeneralizedLinear<float>{
 public:
  ModelAbsoluteRegressionFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelAbsoluteRegression<float> ModelAbsoluteRegressionFloat;
