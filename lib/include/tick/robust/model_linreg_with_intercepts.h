#ifndef LIB_INCLUDE_TICK_ROBUST_MODEL_LINREG_WITH_INTERCEPTS_H_
#define LIB_INCLUDE_TICK_ROBUST_MODEL_LINREG_WITH_INTERCEPTS_H_

// License: BSD 3 clause

#include "tick/linear_model/model_linreg.h"
#include "tick/robust/model_generalized_linear_with_intercepts.h"

template <class T>
class DLL_PUBLIC TModelLinRegWithIntercepts
    : virtual public TModelGeneralizedLinearWithIntercepts<T>,
      virtual public TModelLinReg<T> {
 protected:
  using TModelGeneralizedLinearWithIntercepts<T>::n_samples;
  using TModelGeneralizedLinearWithIntercepts<T>::n_features;
  using TModelGeneralizedLinearWithIntercepts<T>::compute_features_norm_sq;
  using TModelGeneralizedLinearWithIntercepts<T>::features_norm_sq;
  using TModelGeneralizedLinearWithIntercepts<T>::grad_i;
  using TModelGeneralizedLinearWithIntercepts<T>::fit_intercept;
  using TModelGeneralizedLinearWithIntercepts<T>::compute_grad_i;
  using TModelGeneralizedLinearWithIntercepts<T>::features;
  using TModelGeneralizedLinearWithIntercepts<T>::grad_i_factor;
  using TModelGeneralizedLinearWithIntercepts<T>::get_features_norm_sq;
  using TModelGeneralizedLinearWithIntercepts<T>::use_intercept;
  using TModelGeneralizedLinearWithIntercepts<T>::get_n_samples;
  using TModelLinReg<T>::ready_lip_consts;
  using TModelLinReg<T>::lip_consts;

 public:
  using TModelGeneralizedLinear<T>::get_class_name;

 public:
  // This exists soley for cereal/swig
  TModelLinRegWithIntercepts()
      : TModelLinRegWithIntercepts(nullptr, nullptr, false) {}

  TModelLinRegWithIntercepts(const std::shared_ptr<BaseArray2d<T> > features,
                             const std::shared_ptr<SArray<T> > labels,
                             const bool fit_intercept, const int n_threads = 1)
      : TModelLabelsFeatures<T>(features, labels),
        TModelGeneralizedLinear<T>(features, labels, fit_intercept, n_threads),
        TModelGeneralizedLinearWithIntercepts<T>(features, labels,
                                                 fit_intercept, n_threads),
        TModelLinReg<T>(features, labels, fit_intercept, n_threads) {}

  virtual ~TModelLinRegWithIntercepts() {}

  void compute_lip_consts() override;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp(
        "ModelGeneralizedLinearWithIntercepts",
        cereal::virtual_base_class<TModelGeneralizedLinearWithIntercepts<T> >(this)));
    ar(cereal::make_nvp("ModelLinReg",
                        cereal::virtual_base_class<TModelLinReg<T> >(this)));
  }

  BoolStrReport compare(const TModelLinRegWithIntercepts<T> &that,
                        std::stringstream &ss) {
    ss << get_class_name() << std::endl;
    bool are_equal =
        TModelGeneralizedLinearWithIntercepts<T>::compare(that, ss) &&
        TModelLinReg<T>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport compare(const TModelLinRegWithIntercepts<T> &that) {
    std::stringstream ss;
    return compare(that, ss);
  }
  BoolStrReport operator==(const TModelLinRegWithIntercepts<T> &that) {
    return TModelLinRegWithIntercepts<T>::compare(that);
  }
};

using ModelLinRegWithIntercepts = TModelLinRegWithIntercepts<double>;
using ModelLinRegWithInterceptsDouble = TModelLinRegWithIntercepts<double>;
using ModelLinRegWithInterceptsFloat = TModelLinRegWithIntercepts<float>;

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLinRegWithInterceptsDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelLinRegWithInterceptsDouble)

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLinRegWithInterceptsFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelLinRegWithInterceptsFloat)

#endif  // LIB_INCLUDE_TICK_ROBUST_MODEL_LINREG_WITH_INTERCEPTS_H_
