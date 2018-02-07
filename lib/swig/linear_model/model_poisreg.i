// License: BSD 3 clause

%{
#include "tick/linear_model/model_poisreg.h"
%}

%include "model_generalized_linear.i";

enum class LinkType {
    identity = 0,
    exponential
};

template <class T>
class TModelPoisReg : public TModelGeneralizedLinear<T> {
 public:
  TModelPoisReg(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const LinkType link_type,
    const bool fit_intercept,
    const int n_threads
  );

  inline void set_link_type(LinkType link_type);
};

%rename(ModelPoisReg) TModelPoisReg<double>;
class ModelPoisReg : public TModelGeneralizedLinear<double> {
 public:
  ModelPoisReg(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const LinkType link_type,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelPoisReg<double> ModelPoisReg;

%rename(ModelPoisRegDouble) TModelPoisReg<double>;
class TModelPoisReg<double> : public TModelGeneralizedLinear<double> {
 public:
  ModelPoisRegDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const LinkType link_type,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelPoisReg<double> ModelPoisRegDouble;

%rename(ModelPoisRegFloat) TModelPoisReg<float>;
class TModelPoisReg<float> : public TModelGeneralizedLinear<float> {
 public:
  ModelPoisRegFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const LinkType link_type,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelPoisReg<float> ModelPoisRegFloat;
