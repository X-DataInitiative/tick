#ifndef LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_SMOOTHED_HINGE_H_
#define LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_SMOOTHED_HINGE_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"
#include "tick/base_model/model_lipschitz.h"

template <class T, class K = T>
class DLL_PUBLIC TModelSmoothedHinge
    : public virtual TModelGeneralizedLinear<T, K>,
      public TModelLipschitz<T, K> {
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

 private:
  T smoothness;

 public:
  // This exists soley for cereal/swig
  TModelSmoothedHinge() : TModelSmoothedHinge<T, K>(nullptr, nullptr, 0) {}

  TModelSmoothedHinge(const std::shared_ptr<BaseArray2d<T> > features,
                      const std::shared_ptr<SArray<T> > labels,
                      const bool fit_intercept, const T smoothness = 1,
                      const int n_threads = 1)
      : TModelLabelsFeatures<T, K>(features, labels),
        TModelGeneralizedLinear<T, K>(features, labels, fit_intercept,
                                      n_threads) {
    set_smoothness(smoothness);
  }

  T loss_i(const ulong i, const Array<K> &coeffs) override;

  T grad_i_factor(const ulong i, const Array<K> &coeffs) override;

  void compute_lip_consts() override;

  T get_smoothness() const { return smoothness; }

  void set_smoothness(T smoothness) {
    if (smoothness <= 1e-2 || smoothness > 1) {
      TICK_ERROR("smoothness should be between 0.01 and 1");
    } else {
      this->smoothness = smoothness;
    }
  }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp(
        "ModelGeneralizedLinear",
        typename cereal::base_class<TModelGeneralizedLinear<T, K> >(this)));
    ar(cereal::make_nvp(
        "ModelLipschitz",
        typename cereal::base_class<TModelLipschitz<T, K> >(this)));
    ar(CEREAL_NVP(smoothness));
  }

  BoolStrReport compare(const TModelSmoothedHinge<T, K> &that,
                        std::stringstream &ss) {
    ss << get_class_name() << std::endl;
    bool are_equal = TModelGeneralizedLinear<T, K>::compare(that, ss) &&
                     TModelLipschitz<T, K>::compare(that, ss) &&
                     TICK_CMP_REPORT(ss, smoothness);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport compare(const TModelSmoothedHinge<T, K> &that) {
    std::stringstream ss;
    return compare(that, ss);
  }
  BoolStrReport operator==(const TModelSmoothedHinge<T, K> &that) {
    return TModelSmoothedHinge<T, K>::compare(that);
  }
};

using ModelSmoothedHinge = TModelSmoothedHinge<double>;

using ModelSmoothedHingeDouble = TModelSmoothedHinge<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelSmoothedHingeDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelSmoothedHingeDouble)

using ModelSmoothedHingeFloat = TModelSmoothedHinge<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelSmoothedHingeFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelSmoothedHingeFloat)

using ModelSmoothedHingeAtomicDouble =
    TModelSmoothedHinge<double, std::atomic<double> >;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelSmoothedHingeAtomicDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelSmoothedHingeAtomicDouble)

using ModelSmoothedHingeAtomicFloat =
    TModelSmoothedHinge<float, std::atomic<float> >;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelSmoothedHingeAtomicFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelSmoothedHingeAtomicFloat)

#endif  // LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_SMOOTHED_HINGE_H_
