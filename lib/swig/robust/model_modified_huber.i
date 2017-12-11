// License: BSD 3 clause

%{
#include "tick/robust/model_modified_huber.h"
%}


class ModelModifiedHuber : public virtual ModelGeneralizedLinear,
                    public ModelLipschitz {
 public:

  ModelModifiedHuber(const SBaseArrayDouble2dPtr features,
              const SArrayDoublePtr labels,
              const bool fit_intercept,
              const int n_threads);
};
