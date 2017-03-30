
%{
#include "model_generalized_linear.h"
%}

class ModelGeneralizedLinear : public ModelLabelsFeatures {
 public:
  ModelGeneralizedLinear(const SBaseArrayDouble2dPtr features,
                         const SArrayDoublePtr labels,
                         const bool fit_intercept,
                         const int n_threads = 1);

  unsigned long get_n_coeffs() const override;

  virtual void set_fit_intercept(bool fit_intercept);
};
