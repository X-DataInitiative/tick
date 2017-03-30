%{
#include "linreg_with_intercepts.h"
%}


class ModelLinRegWithIntercepts : public ModelGeneralizedLinearWithIntercepts,
                                  public ModelLipschitz {
 public:

  ModelLinRegWithIntercepts(const SBaseArrayDouble2dPtr features,
                            const SArrayDoublePtr labels,
                            const int n_threads);
};
