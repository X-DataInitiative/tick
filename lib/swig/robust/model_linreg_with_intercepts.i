// License: BSD 3 clause

%{
#include "tick/robust/model_linreg_with_intercepts.h"
%}

%include "model_linreg.i"
%include "model_generalized_linear_with_intercepts.i"

template <class T>
class TModelLinRegWithIntercepts : public virtual TModelGeneralizedLinearWithIntercepts<T, K>,
                    public TModelLinReg<T, K> {
 public:
  TModelLinRegWithIntercepts();
  TModelLinRegWithIntercepts(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads
  );
  bool compare(const TModelLinRegWithIntercepts<T, K> &that);
};

%rename(ModelLinRegWithInterceptsDouble) TModelLinRegWithIntercepts<double, double>;
class ModelLinRegWithInterceptsDouble : public virtual ModelGeneralizedLinearWithInterceptsDouble,
                    public ModelLinRegDouble {
 public:
  ModelLinRegWithInterceptsDouble();
  ModelLinRegWithInterceptsDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
  bool compare(const ModelLinRegWithInterceptsDouble &that);
};
typedef TModelLinRegWithIntercepts<double, double> ModelLinRegWithInterceptsDouble;
TICK_MAKE_PICKLABLE(ModelLinRegWithInterceptsDouble);

%rename(ModelLinRegWithInterceptsFloat) TModelLinRegWithIntercepts<float, float>;
class ModelLinRegWithInterceptsFloat : public virtual ModelGeneralizedLinearWithInterceptsFloat,
                    public ModelLinRegFloat {
 public:
  ModelLinRegWithInterceptsFloat();
  ModelLinRegWithInterceptsFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
  bool compare(const ModelLinRegWithInterceptsFloat &that);
};
typedef TModelLinRegWithIntercepts<float, float> ModelLinRegWithInterceptsFloat;
TICK_MAKE_PICKLABLE(ModelLinRegWithInterceptsFloat);
