// License: BSD 3 clause

%{
#include "tick/robust/model_linreg_with_intercepts.h"
%}

%include "model_linreg.i"
%include "model_generalized_linear_with_intercepts.i"

template <class T>
class TModelLinRegWithIntercepts : public virtual TModelGeneralizedLinearWithIntercepts<T>,
                    public TModelLinReg<T> {
 public:
  TModelLinRegWithIntercepts(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads
  );
};

%rename(ModelLinRegWithIntercepts) TModelLinRegWithIntercepts<double>;
class ModelLinRegWithIntercepts : public virtual ModelGeneralizedLinearWithInterceptsDouble,
                    public TModelLinReg<double> {
 public:
  ModelLinRegWithIntercepts(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelLinRegWithIntercepts<double> ModelLinRegWithIntercepts;

%rename(ModelLinRegWithInterceptsDouble) TModelLinRegWithIntercepts<double>;
class ModelLinRegWithInterceptsDouble : public virtual ModelGeneralizedLinearWithInterceptsDouble,
                    public TModelLinReg<double> {
 public:
  ModelLinRegWithInterceptsDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelLinRegWithIntercepts<double> ModelLinRegWithInterceptsDouble;

%rename(ModelLinRegWithInterceptsFloat) TModelLinRegWithIntercepts<float>;
class ModelLinRegWithInterceptsFloat : public virtual ModelGeneralizedLinearWithInterceptsFloat,
                    public TModelLinReg<float> {
 public:
  ModelLinRegWithInterceptsFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelLinRegWithIntercepts<float> ModelLinRegWithInterceptsFloat;
