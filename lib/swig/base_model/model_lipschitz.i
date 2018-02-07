// License: BSD 3 clause


%{
#include "tick/base_model/model_lipschitz.h"
%}

%include "model.i"

// An interface for a Model with the ability to compute Lipschitz constants

template <class T>
class TModelLipschitz : public virtual TModel<T> {
 public:
  TModelLipschitz();
  T get_lip_max() override;
  T get_lip_mean() override;
};

%template(ModelLipschitz) TModelLipschitz<double>;
typedef TModelLipschitz<double> ModelLipschitz;

%template(ModelLipschitzDouble) TModelLipschitz<double>;
typedef TModelLipschitz<double> ModelLipschitzDouble;

%template(ModelLipschitzFloat) TModelLipschitz<float>;
typedef TModelLipschitz<float> ModelLipschitzFloat;
