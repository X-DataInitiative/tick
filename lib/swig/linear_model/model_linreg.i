// License: BSD 3 clause

%{
#include "tick/linear_model/model_linreg.h"
%}


class ModelLinReg : public virtual ModelGeneralizedLinear,
                    public ModelLipschitz {
 public:

  ModelLinReg(const SBaseArrayDouble2dPtr features,
              const SArrayDoublePtr labels,
              const bool fit_intercept,
              const int n_threads);
};
