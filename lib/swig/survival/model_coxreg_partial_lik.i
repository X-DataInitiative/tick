// License: BSD 3 clause

%{
#include "tick/survival/model_coxreg_partial_lik.h"
%}

%include "model.i"

class ModelCoxRegPartialLik : public TModel<double, double> {
 public:

  ModelCoxRegPartialLik(const SBaseArrayDouble2dPtr features,
                        const SArrayDoublePtr times,
                        const SArrayUShortPtr censoring);
};
