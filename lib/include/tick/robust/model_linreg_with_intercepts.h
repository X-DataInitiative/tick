#ifndef LIB_INCLUDE_TICK_ROBUST_MODEL_LINREG_WITH_INTERCEPTS_H_
#define LIB_INCLUDE_TICK_ROBUST_MODEL_LINREG_WITH_INTERCEPTS_H_

// License: BSD 3 clause

#include "tick/linear_model/model_linreg.h"
#include "tick/robust/model_generalized_linear_with_intercepts.h"

template <class T, class K = T>
class DLL_PUBLIC TModelLinRegWithIntercepts
    : virtual public TModelGeneralizedLinearWithIntercepts<T, K>,
      virtual public TModelLinReg<T, K> {
 protected:
  using TModelGeneralizedLinearWithIntercepts<T, K>::n_samples;
  using TModelGeneralizedLinearWithIntercepts<T, K>::n_features;
  using TModelGeneralizedLinearWithIntercepts<T, K>::compute_features_norm_sq;
  using TModelGeneralizedLinearWithIntercepts<T, K>::features_norm_sq;
  using TModelGeneralizedLinearWithIntercepts<T, K>::grad_i;
  using TModelGeneralizedLinearWithIntercepts<T, K>::fit_intercept;
  using TModelGeneralizedLinearWithIntercepts<T, K>::compute_grad_i;
  using TModelGeneralizedLinearWithIntercepts<T, K>::features;
  using TModelGeneralizedLinearWithIntercepts<T, K>::grad_i_factor;
  using TModelGeneralizedLinearWithIntercepts<T, K>::get_features_norm_sq;
  using TModelGeneralizedLinearWithIntercepts<T, K>::use_intercept;
  using TModelGeneralizedLinearWithIntercepts<T, K>::get_n_samples;
  using TModelLinReg<T, K>::ready_lip_consts;
  using TModelLinReg<T, K>::lip_consts;

 public:
  using TModelGeneralizedLinear<T, K>::get_class_name;

 public:
  // This exists soley for cereal/swig
  TModelLinRegWithIntercepts()
      : TModelLinRegWithIntercepts(nullptr, nullptr, false) {}

  TModelLinRegWithIntercepts(const std::shared_ptr<BaseArray2d<T> > features,
                             const std::shared_ptr<SArray<T> > labels,
                             const bool fit_intercept, const int n_threads = 1)
      : TModelLabelsFeatures<T, K>(features, labels),
        TModelGeneralizedLinear<T, K>(features, labels, fit_intercept,
                                      n_threads),
        TModelGeneralizedLinearWithIntercepts<T, K>(features, labels,
                                                    fit_intercept, n_threads),
        TModelLinReg<T, K>(features, labels, fit_intercept, n_threads) {}

  virtual ~TModelLinRegWithIntercepts() {}

  void compute_lip_consts() override;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp(
        "ModelGeneralizedLinearWithIntercepts",
        cereal::virtual_base_class<
            TModelGeneralizedLinearWithIntercepts<T, K> >(this)));
    ar(cereal::make_nvp("ModelLinReg",
                        cereal::virtual_base_class<TModelLinReg<T, K> >(this)));
  }

  BoolStrReport compare(const TModelLinRegWithIntercepts<T, K> &that,
                        std::stringstream &ss) {
    ss << get_class_name() << std::endl;
    bool are_equal =
        TModelGeneralizedLinearWithIntercepts<T, K>::compare(that, ss) &&
        TModelLinReg<T, K>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport compare(const TModelLinRegWithIntercepts<T, K> &that) {
    std::stringstream ss;
    return compare(that, ss);
  }
  BoolStrReport operator==(const TModelLinRegWithIntercepts<T, K> &that) {
    return TModelLinRegWithIntercepts<T, K>::compare(that);
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
