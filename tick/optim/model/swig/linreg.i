%{
#include "linreg.h"
%}


class ModelLinReg : public ModelGeneralizedLinear,
                    public ModelLipschitz {
 public:

  ModelLinReg(const SBaseArrayDouble2dPtr features,
              const SArrayDoublePtr labels,
              const bool fit_intercept,
              const int n_threads);

};
