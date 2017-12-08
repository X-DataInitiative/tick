// License: BSD 3 clause

%{
#include "tick/robust/model_absolute_regression.h"
%}


class ModelAbsoluteRegression : public virtual ModelGeneralizedLinear {
 public:
  ModelAbsoluteRegression(const SBaseArrayDouble2dPtr features,
                          const SArrayDoublePtr labels,
                          const bool fit_intercept,
                          const int n_threads = 1);
};
