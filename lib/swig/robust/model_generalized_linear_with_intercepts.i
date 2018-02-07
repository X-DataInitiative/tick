// License: BSD 3 clause

%{
#include "tick/base_model/model_generalized_linear.h"
%}

%include "model_generalized_linear.i"

template <class T>
class TModelGeneralizedLinearWithIntercepts : public virtual TModelGeneralizedLinear<T> {
 public:
  TModelGeneralizedLinearWithIntercepts(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads
  );
};

%rename(ModelGeneralizedLinearWithIntercepts) TModelGeneralizedLinearWithIntercepts<double>;
class ModelGeneralizedLinearWithIntercepts : public virtual TModelGeneralizedLinear<double> {
 public:
  ModelGeneralizedLinearWithIntercepts(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelGeneralizedLinearWithIntercepts<double> ModelGeneralizedLinearWithIntercepts;

%rename(ModelGeneralizedLinearWithInterceptsDouble) TModelGeneralizedLinearWithIntercepts<double>;
class ModelGeneralizedLinearWithInterceptsDouble : public virtual TModelGeneralizedLinear<double> {
 public:
  ModelGeneralizedLinearWithInterceptsDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelGeneralizedLinearWithIntercepts<double> ModelGeneralizedLinearWithInterceptsDouble;

%rename(ModelGeneralizedLinearWithInterceptsFloat) TModelGeneralizedLinearWithIntercepts<float>;
class ModelGeneralizedLinearWithInterceptsFloat : public virtual TModelGeneralizedLinear<float> {
 public:
  ModelGeneralizedLinearWithInterceptsFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelGeneralizedLinearWithIntercepts<float> ModelGeneralizedLinearWithInterceptsFloat;
