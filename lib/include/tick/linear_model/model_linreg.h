//
// Created by Stéphane GAIFFAS on 12/12/2015.
//

#ifndef LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_LINREG_H_
#define LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_LINREG_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"
#include "tick/base_model/model_lipschitz.h"

template <class T, class K = T>
class DLL_PUBLIC TModelLinReg : public virtual TModelGeneralizedLinear<T, K>,
                                public TModelLipschitz<T, K> {
 protected:
  using TModelLipschitz<T, K>::ready_lip_consts;
  using TModelLipschitz<T, K>::lip_consts;
  using TModelGeneralizedLinear<T, K>::compute_features_norm_sq;
  using TModelGeneralizedLinear<T, K>::n_samples;
  using TModelGeneralizedLinear<T, K>::features_norm_sq;
  using TModelGeneralizedLinear<T, K>::fit_intercept;

 public:
  using TModelGeneralizedLinear<T, K>::get_label;
  using TModelGeneralizedLinear<T, K>::use_intercept;
  using TModelGeneralizedLinear<T, K>::get_inner_prod;
  using TModelGeneralizedLinear<T, K>::get_class_name;

 public:
  // This exists soley for cereal/swig
  TModelLinReg() : TModelLinReg<T, K>(nullptr, nullptr, 0, 0) {}

  TModelLinReg(const std::shared_ptr<BaseArray2d<T> > features,
               const std::shared_ptr<SArray<T> > labels,
               const bool fit_intercept, const int n_threads = 1)
      : TModelLabelsFeatures<T, K>(features, labels),
        TModelGeneralizedLinear<T, K>(features, labels, fit_intercept,
                                      n_threads) {}

  virtual ~TModelLinReg() {}

  T sdca_dual_min_i(const ulong i, const T dual_i,
                    const Array<K> &primal_vector,
                    const T previous_delta_dual_i, T l_l2sq) override;

  T loss_i(const ulong i, const Array<K> &coeffs) override;

  T grad_i_factor(const ulong i, const Array<K> &coeffs) override;

  void compute_lip_consts() override;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp(
        "ModelGeneralizedLinear",
        typename cereal::virtual_base_class<TModelGeneralizedLinear<T, K> >(
            this)));
    ar(cereal::make_nvp(
        "ModelLipschitz",
        typename cereal::base_class<TModelLipschitz<T, K> >(this)));
  }

  BoolStrReport compare(const TModelLinReg<T, K> &that, std::stringstream &ss) {
    ss << get_class_name() << std::endl;
    bool are_equal = TModelGeneralizedLinear<T, K>::compare(that, ss) &&
                     TModelLipschitz<T, K>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport compare(const TModelLinReg<T, K> &that) {
    std::stringstream ss;
    return compare(that, ss);
  }
  BoolStrReport operator==(const TModelLinReg<T, K> &that) {
    return TModelLinReg<T, K>::compare(that);
  }
};

using ModelLinReg = TModelLinReg<double, double>;

using ModelLinRegDouble = TModelLinReg<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLinRegDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelLinRegDouble)

using ModelLinRegFloat = TModelLinReg<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLinRegFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelLinRegFloat)

using ModelLinRegAtomicDouble = TModelLinReg<double, std::atomic<double> >;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLinRegAtomicDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelLinRegAtomicDouble)

using ModelLinRegAtomicFloat = TModelLinReg<float, std::atomic<float> >;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLinRegAtomicFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelLinRegAtomicFloat)

#endif  // LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_LINREG_H_
