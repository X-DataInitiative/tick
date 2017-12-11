// License: BSD 3 clause

%{
#include "tick/linear_model/model_quadratic_hinge.h"
%}


class ModelQuadraticHinge : public virtual ModelGeneralizedLinear,
                    public ModelLipschitz {
 public:

  ModelQuadraticHinge(const SBaseArrayDouble2dPtr features,
              const SArrayDoublePtr labels,
              const bool fit_intercept,
              const int n_threads);
};
