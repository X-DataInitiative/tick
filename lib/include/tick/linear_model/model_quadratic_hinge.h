#ifndef LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_QUADRATIC_HINGE_H_
#define LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_QUADRATIC_HINGE_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"
#include "tick/base_model/model_lipschitz.h"

template <class T, class K = T>
class DLL_PUBLIC TModelQuadraticHinge
    : public virtual TModelGeneralizedLinear<T, K>,
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
  TModelQuadraticHinge()
      : TModelQuadraticHinge<T, K>(nullptr, nullptr, false) {}

  TModelQuadraticHinge(const std::shared_ptr<BaseArray2d<T>> features,
                       const std::shared_ptr<SArray<T>> labels,
                       const bool fit_intercept, const int n_threads = 1)
      : TModelLabelsFeatures<T, K>(features, labels),
        TModelGeneralizedLinear<T, K>(features, labels, fit_intercept,
                                      n_threads) {}

  T loss_i(const ulong i, const Array<K> &coeffs) override;

  T grad_i_factor(const ulong i, const Array<K> &coeffs) override;

  void compute_lip_consts() override;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp(
        "ModelGeneralizedLinear",
        typename cereal::virtual_base_class<TModelGeneralizedLinear<T, K>>(
            this)));
    ar(cereal::make_nvp(
        "ModelLipschitz",
        typename cereal::base_class<TModelLipschitz<T, K>>(this)));
  }

  BoolStrReport compare(const TModelQuadraticHinge<T, K> &that,
                        std::stringstream &ss) {
    ss << get_class_name() << std::endl;
    bool are_equal = TModelGeneralizedLinear<T, K>::compare(that, ss) &&
                     TModelLipschitz<T, K>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport compare(const TModelQuadraticHinge<T, K> &that) {
    std::stringstream ss;
    return compare(that, ss);
  }
  BoolStrReport operator==(const TModelQuadraticHinge<T, K> &that) {
    return TModelQuadraticHinge<T, K>::compare(that);
  }
};

using ModelQuadraticHinge = TModelQuadraticHinge<double>;

using ModelQuadraticHingeDouble = TModelQuadraticHinge<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelQuadraticHingeDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelQuadraticHingeDouble)

using ModelQuadraticHingeFloat = TModelQuadraticHinge<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelQuadraticHingeFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelQuadraticHingeFloat)

using ModelQuadraticHingeAtomicDouble =
    TModelQuadraticHinge<double, std::atomic<double>>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelQuadraticHingeAtomicDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelQuadraticHingeAtomicDouble)

using ModelQuadraticHingeAtomicFloat =
    TModelQuadraticHinge<float, std::atomic<float>>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelQuadraticHingeAtomicFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelQuadraticHingeAtomicFloat)

#endif  // LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_QUADRATIC_HINGE_H_
