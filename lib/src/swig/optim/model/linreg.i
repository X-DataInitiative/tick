// License: BSD 3 clause

%{
#include "tick/optim/model/linreg.h"
%}


class ModelLinReg : public virtual ModelGeneralizedLinear,
                    public ModelLipschitz {
 public:
  // Empty constructor only used for serialization
  ModelLinReg();

  ModelLinReg(const SBaseArrayDouble2dPtr features,
              const SArrayDoublePtr labels,
              const bool fit_intercept,
              const int n_threads);
};

TICK_MAKE_PICKLABLE(ModelLinReg);