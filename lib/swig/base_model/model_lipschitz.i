// License: BSD 3 clause


%{
#include "tick/base_model/model_lipschitz.h"
%}


// An interface for a Model with the ability to compute Lipschitz constants
class ModelLipschitz : public virtual Model {
 public:

  ModelLipschitz();

  double get_lip_max() override;
  double get_lip_mean() override;
};
