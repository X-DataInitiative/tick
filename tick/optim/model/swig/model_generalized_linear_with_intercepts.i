
%{
#include "model_generalized_linear.h"
%}

class ModelGeneralizedLinearWithIntercepts : public ModelGeneralizedLinear {
 public:
  ModelGeneralizedLinearWithIntercepts(const SBaseArrayDouble2dPtr features,
                                       const SArrayDoublePtr labels,
                                       const int n_threads = 1);

  unsigned long get_n_coeffs() const override;
};
