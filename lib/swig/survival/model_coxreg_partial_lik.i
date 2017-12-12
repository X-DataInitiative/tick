// License: BSD 3 clause

%{
#include "tick/survival/model_coxreg_partial_lik.h"
%}


class ModelCoxRegPartialLik : public Model {
 public:

  ModelCoxRegPartialLik(const SBaseArrayDouble2dPtr features,
                        const SArrayDoublePtr times,
                        const SArrayUShortPtr censoring);
};
