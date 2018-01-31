// License: BSD 3 clause

%{
#include "tick/base_model/model_generalized_linear.h"
%}

%include "model_generalized_linear.i"

template <class T, class K>
class TModelGeneralizedLinearWithIntercepts : public virtual TModelGeneralizedLinear<T, K>{
 public:
  TModelGeneralizedLinearWithIntercepts(
    const std::shared_ptr<BaseArray2d<K> > features,
    const std::shared_ptr<SArray<K> > labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};

%template(TModelGeneralizedLinearWithInterceptsDouble) TModelGeneralizedLinearWithIntercepts<double, double>;
typedef TModelGeneralizedLinearWithIntercepts<double, double> TModelGeneralizedLinearWithInterceptsDouble;

class ModelGeneralizedLinearWithIntercepts : public TModelGeneralizedLinearWithInterceptsDouble {
 public:
  ModelGeneralizedLinearWithIntercepts(const SBaseArrayDouble2dPtr features,
                                       const SArrayDoublePtr labels,
                                       const bool fit_intercept,
                                       const int n_threads = 1);
};
