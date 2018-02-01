// License: BSD 3 clause


%{
#include "tick/base_model/model_lipschitz.h"
%}

%include "model.i"

// An interface for a Model with the ability to compute Lipschitz constants

template <class T, class K = T>
class TModelLipschitz : public virtual TModel<K, T> {
 public:
  TModelLipschitz();
  K get_lip_max() override;
  K get_lip_mean() override;
};

%template(ModelLipschitzDouble) TModelLipschitz<double, double>;
typedef TModelLipschitz<double, double> ModelLipschitzDouble;

%template(ModelLipschitzFloat) TModelLipschitz<float, float>;
typedef TModelLipschitz<float, float> ModelLipschitzFloat;

class ModelLipschitz : public virtual ModelLipschitzDouble {
 public:
  ModelLipschitz();
  double get_lip_max() override;
  double get_lip_mean() override;
};
