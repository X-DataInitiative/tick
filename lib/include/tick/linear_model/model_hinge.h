#ifndef LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_HINGE_H_
#define LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_HINGE_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"

template <class T, class K = T>
class DLL_PUBLIC TModelHinge : public virtual TModelGeneralizedLinear<T, K> {
 protected:
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
  TModelHinge() : TModelHinge<T, K>(nullptr, nullptr, false) {}

  TModelHinge(const std::shared_ptr<BaseArray2d<T> > features,
              const std::shared_ptr<SArray<T> > labels,
              const bool fit_intercept, const int n_threads = 1)
      : TModelLabelsFeatures<T, K>(features, labels),
        TModelGeneralizedLinear<T, K>(features, labels, fit_intercept,
                                      n_threads) {}

  T loss_i(const ulong i, const Array<K> &coeffs) override;

  T grad_i_factor(const ulong i, const Array<K> &coeffs) override;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp(
        "ModelGeneralizedLinear",
        typename cereal::base_class<TModelGeneralizedLinear<T, K> >(this)));
  }

  BoolStrReport compare(const TModelHinge<T, K> &that, std::stringstream &ss) {
    ss << get_class_name() << std::endl;
    return TModelGeneralizedLinear<T, K>::compare(that, ss);
  }
  BoolStrReport compare(const TModelHinge<T, K> &that) {
    std::stringstream ss;
    return compare(that, ss);
  }
  BoolStrReport operator==(const TModelHinge<T, K> &that) {
    return TModelHinge<T, K>::compare(that);
  }
};

using ModelHinge = TModelHinge<double>;

using ModelHingeDouble = TModelHinge<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelHingeDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelHingeDouble)

using ModelHingeFloat = TModelHinge<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelHingeFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelHingeFloat)

using ModelHingeAtomicDouble = TModelHinge<double, std::atomic<double> >;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelHingeAtomicDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelHingeAtomicDouble)

using ModelHingeAtomicFloat = TModelHinge<float, std::atomic<float> >;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelHingeAtomicFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelHingeAtomicFloat)

#endif  // LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_HINGE_H_
