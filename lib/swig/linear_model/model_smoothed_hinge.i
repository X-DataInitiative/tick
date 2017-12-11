// License: BSD 3 clause

%{
#include "tick/linear_model/model_smoothed_hinge.h"
%}


class ModelSmoothedHinge : public virtual ModelGeneralizedLinear,
                    public ModelLipschitz {
 public:

  ModelSmoothedHinge(const SBaseArrayDouble2dPtr features,
              const SArrayDoublePtr labels,
              const bool fit_intercept,
              const double smoothness,
              const int n_threads);

  double get_smoothness() const;

  void set_smoothness(double smoothness);
};
