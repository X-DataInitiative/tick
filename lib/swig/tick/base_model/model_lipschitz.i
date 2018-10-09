// License: BSD 3 clause


%{
#include "tick/base_model/model_lipschitz.h"
%}

%include "model.i"

// An interface for a Model with the ability to compute Lipschitz constants

template <class T, class K>
class TModelLipschitz : public virtual TModel<T, K> {
 public:
  TModelLipschitz();
  T get_lip_max() override;
  T get_lip_mean() override;
};

%template(ModelLipschitz) TModelLipschitz<double, double>;
typedef TModelLipschitz<double, double> ModelLipschitz;

%template(ModelLipschitzDouble) TModelLipschitz<double, double>;
typedef TModelLipschitz<double, double> ModelLipschitzDouble;

%template(ModelLipschitzFloat) TModelLipschitz<float, float>;
typedef TModelLipschitz<float, float> ModelLipschitzFloat;

%template(ModelLipschitzAtomicDouble) TModelLipschitz<double, std::atomic<double> >;
typedef TModelLipschitz<double, std::atomic<double> > ModelLipschitzAtomicDouble;

%template(ModelLipschitzAtomicFloat) TModelLipschitz<float, std::atomic<float> >;
typedef TModelLipschitz<float, std::atomic<float> > ModelLipschitzAtomicFloat;
