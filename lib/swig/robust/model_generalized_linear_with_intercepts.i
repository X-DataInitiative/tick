// License: BSD 3 clause


%{
#include "tick/base_model/model_generalized_linear.h"
%}

class ModelGeneralizedLinearWithIntercepts : public virtual ModelGeneralizedLinear {
 public:
  ModelGeneralizedLinearWithIntercepts(const SBaseArrayDouble2dPtr features,
                                       const SArrayDoublePtr labels,
                                       const bool fit_intercept,
                                       const int n_threads = 1);
};
