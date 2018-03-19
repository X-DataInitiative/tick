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
  TModelPoisReg();
  TModelPoisReg(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const LinkType link_type,
    const bool fit_intercept,
    const int n_threads
  );

  inline void set_link_type(LinkType link_type);

  bool compare(const TModelPoisReg<T> &that);
};

%rename(ModelPoisRegDouble) TModelPoisReg<double>;
class ModelPoisRegDouble : public TModelGeneralizedLinear<double> {
 public:
  ModelPoisRegDouble();
  ModelPoisRegDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const LinkType link_type,
    const bool fit_intercept,
    const int n_threads = 1
  );

  bool compare(const ModelPoisRegDouble &that);
};
typedef TModelPoisReg<double> ModelPoisRegDouble;
TICK_MAKE_PICKLABLE(ModelPoisRegDouble);

%rename(ModelPoisRegFloat) TModelPoisReg<float>;
class ModelPoisRegFloat : public TModelGeneralizedLinear<float> {
 public:
  ModelPoisRegFloat();
  ModelPoisRegFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const LinkType link_type,
    const bool fit_intercept,
    const int n_threads = 1
  );

  bool compare(const ModelPoisRegFloat &that);
};
typedef TModelPoisReg<float> ModelPoisRegFloat;
TICK_MAKE_PICKLABLE(ModelPoisRegFloat);
