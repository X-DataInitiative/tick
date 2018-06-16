// License: BSD 3 clause

%{
#include "tick/base_model/model_generalized_linear.h"
%}

%include "model_generalized_linear.i"

template <class T>
class TModelGeneralizedLinearWithIntercepts : public virtual TModelGeneralizedLinear<T, K> {
 public:
  TModelGeneralizedLinearWithIntercepts();
  TModelGeneralizedLinearWithIntercepts(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads
  );
  bool compare(const TModelGeneralizedLinearWithIntercepts<T, K> &that);
};

%rename(ModelGeneralizedLinearWithInterceptsDouble) TModelGeneralizedLinearWithIntercepts<double, double>;
class ModelGeneralizedLinearWithInterceptsDouble : public virtual TModelGeneralizedLinear<double, double> {
 public:
  ModelGeneralizedLinearWithInterceptsDouble();
  ModelGeneralizedLinearWithInterceptsDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
  bool compare(const ModelGeneralizedLinearWithInterceptsDouble &that);
};
typedef TModelGeneralizedLinearWithIntercepts<double, double> ModelGeneralizedLinearWithInterceptsDouble;
TICK_MAKE_PICKLABLE(ModelGeneralizedLinearWithInterceptsDouble);

%rename(ModelGeneralizedLinearWithInterceptsFloat) TModelGeneralizedLinearWithIntercepts<float, float>;
class ModelGeneralizedLinearWithInterceptsFloat : public virtual TModelGeneralizedLinear<float, float> {
 public:
  ModelGeneralizedLinearWithInterceptsFloat();
  ModelGeneralizedLinearWithInterceptsFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
  bool compare(const ModelGeneralizedLinearWithInterceptsFloat &that);
};
typedef TModelGeneralizedLinearWithIntercepts<float, float> ModelGeneralizedLinearWithInterceptsFloat;
TICK_MAKE_PICKLABLE(ModelGeneralizedLinearWithInterceptsFloat);
