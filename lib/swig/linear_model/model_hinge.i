// License: BSD 3 clause

%{
#include "tick/linear_model/model_hinge.h"
%}


class ModelHinge : public virtual ModelGeneralizedLinear {
 public:

  ModelHinge(const SBaseArrayDouble2dPtr features,
              const SArrayDoublePtr labels,
              const bool fit_intercept,
              const int n_threads);
};
