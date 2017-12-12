// License: BSD 3 clause

%{
#include "tick/linear_model/model_poisreg.h"
%}

enum class LinkType {
    identity = 0,
    exponential
};

class ModelPoisReg : public ModelGeneralizedLinear {
 public:

  ModelPoisReg(const SBaseArrayDouble2dPtr features,
               const SArrayDoublePtr labels,
               const LinkType link_type,
               const bool fit_intercept,
               const int n_threads);

  inline void set_link_type(LinkType link_type);
};
