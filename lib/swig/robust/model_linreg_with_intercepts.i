// License: BSD 3 clause

%{
#include "tick/robust/model_linreg_with_intercepts.h"
%}


class ModelLinRegWithIntercepts : public ModelGeneralizedLinearWithIntercepts,
                                  public ModelLinReg {
 public:

  ModelLinRegWithIntercepts(const SBaseArrayDouble2dPtr features,
                            const SArrayDoublePtr labels,
                            const bool fit_intercept,
                            const int n_threads);
};
