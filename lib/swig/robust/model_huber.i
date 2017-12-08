// License: BSD 3 clause

%{
#include "tick/robust/model_huber.h"
%}


class ModelHuber : public virtual ModelGeneralizedLinear,
                    public ModelLipschitz {
 public:

  ModelHuber(const SBaseArrayDouble2dPtr features,
              const SArrayDoublePtr labels,
              const bool fit_intercept,
              const double threshold,
              const int n_threads);

  virtual double get_threshold(void) const;

  virtual void set_threshold(const double threshold);
};
