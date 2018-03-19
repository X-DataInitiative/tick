// License: BSD 3 clause

%{
#include "tick/robust/model_absolute_regression.h"
%}

template <class T>
class TModelAbsoluteRegression : public virtual TModelGeneralizedLinear<T>{
 public:
  TModelAbsoluteRegression();
  TModelAbsoluteRegression(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
  bool compare(const TModelAbsoluteRegression<T> &that);
};

%rename(ModelAbsoluteRegressionDouble) TModelAbsoluteRegression<double>;
class ModelAbsoluteRegressionDouble : public virtual TModelGeneralizedLinear<double>{
 public:
  ModelAbsoluteRegressionDouble();
  ModelAbsoluteRegressionDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
  bool compare(const ModelAbsoluteRegressionDouble &that);
};
typedef TModelAbsoluteRegression<double> ModelAbsoluteRegressionDouble;
TICK_MAKE_PICKLABLE(ModelAbsoluteRegressionDouble);

%rename(ModelAbsoluteRegressionFloat) TModelAbsoluteRegression<float>;
class ModelAbsoluteRegressionFloat : public virtual TModelGeneralizedLinear<float>{
 public:
  ModelAbsoluteRegressionFloat();
  ModelAbsoluteRegressionFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
  bool compare(const ModelAbsoluteRegressionFloat &that);
};
typedef TModelAbsoluteRegression<float> ModelAbsoluteRegressionFloat;
TICK_MAKE_PICKLABLE(ModelAbsoluteRegressionFloat);
