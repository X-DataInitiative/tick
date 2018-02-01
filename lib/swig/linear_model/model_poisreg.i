// License: BSD 3 clause

%{
#include "tick/linear_model/model_poisreg.h"
%}

%include "model_generalized_linear.i";

enum class LinkType {
    identity = 0,
    exponential
};

template <class T, class K>
class TModelPoisReg : public TModelGeneralizedLinear<T, K> {
 public:

  ModelPoisReg(const SBaseArrayDouble2dPtr features,
               const SArrayDoublePtr labels,
               const LinkType link_type,
               const bool fit_intercept,
               const int n_threads);

  inline void set_link_type(LinkType link_type);
};

%rename(ModelPoisRegDouble) TModelPoisReg<double, double>;
class TModelPoisReg<double, double> : public TModelGeneralizedLinear<double, double> {
 public:
  TModelPoisReg(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const LinkType link_type,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelPoisReg<double, double> ModelPoisRegDouble;

%rename(ModelPoisRegFloat) TModelPoisReg<float, float>;
class TModelPoisReg<float, float> : public TModelGeneralizedLinear<float, float> {
 public:
  TModelPoisReg(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const LinkType link_type,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelPoisReg<float, float> ModelPoisRegFloat;

class ModelPoisReg : public ModelPoisRegDouble {
 public:
  ModelPoisReg(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const LinkType link_type,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
