// License: BSD 3 clause

%{
#include "tick/robust/model_epsilon_insensitive.h"
%}


class ModelEpsilonInsensitive : public virtual ModelGeneralizedLinear {
 public:

  ModelEpsilonInsensitive(const SBaseArrayDouble2dPtr features,
              const SArrayDoublePtr labels,
              const bool fit_intercept,
              const double threshold,
              const int n_threads);

  virtual double get_threshold(void) const;

  virtual void set_threshold(const double threshold);
};
