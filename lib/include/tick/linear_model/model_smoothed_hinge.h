#ifndef LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_SMOOTHED_HINGE_H_
#define LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_SMOOTHED_HINGE_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"
#include "tick/base_model/model_lipschitz.h"

template <class T>
class DLL_PUBLIC TModelSmoothedHinge
    : public virtual TModelGeneralizedLinear<T>,
      public TModelLipschitz<T> {
 protected:
  using TModelLipschitz<T>::ready_lip_consts;
  using TModelLipschitz<T>::lip_consts;
  using TModelGeneralizedLinear<T>::compute_features_norm_sq;
  using TModelGeneralizedLinear<T>::n_samples;
  using TModelGeneralizedLinear<T>::features_norm_sq;
  using TModelGeneralizedLinear<T>::fit_intercept;

 public:
  using TModelGeneralizedLinear<T>::get_label;
  using TModelGeneralizedLinear<T>::use_intercept;
  using TModelGeneralizedLinear<T>::get_inner_prod;
  using TModelGeneralizedLinear<T>::get_class_name;

 private:
  T smoothness;

 public:
  // This exists soley for cereal/swig
  TModelSmoothedHinge() : TModelSmoothedHinge<T>(nullptr, nullptr, 0) {}

  TModelSmoothedHinge(const std::shared_ptr<BaseArray2d<T> > features,
                      const std::shared_ptr<SArray<T> > labels,
                      const bool fit_intercept, const T smoothness = 1,
                      const int n_threads = 1)
      : TModelLabelsFeatures<T>(features, labels),
        TModelGeneralizedLinear<T>(features, labels, fit_intercept, n_threads) {
    set_smoothness(smoothness);
  }

  T loss_i(const ulong i, const Array<T> &coeffs) override;

  T grad_i_factor(const ulong i, const Array<T> &coeffs) override;

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
        typename cereal::base_class<TModelGeneralizedLinear<T> >(this)));
    ar(cereal::make_nvp(
        "ModelLipschitz",
        typename cereal::base_class<TModelLipschitz<T> >(this)));
    ar(CEREAL_NVP(smoothness));
  }

  BoolStrReport compare(const TModelSmoothedHinge<T> &that,
                        std::stringstream &ss) {
    ss << get_class_name() << std::endl;
    bool are_equal = TModelGeneralizedLinear<T>::compare(that, ss) &&
                     TModelLipschitz<T>::compare(that, ss) &&
                     TICK_CMP_REPORT(ss, smoothness);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport compare(const TModelSmoothedHinge<T> &that) {
    std::stringstream ss;
    return compare(that, ss);
  }
  BoolStrReport operator==(const TModelSmoothedHinge<T> &that) {
    return TModelSmoothedHinge<T>::compare(that);
  }
};

using ModelSmoothedHinge = TModelSmoothedHinge<double>;
using ModelSmoothedHingeDouble = TModelSmoothedHinge<double>;
using ModelSmoothedHingeFloat = TModelSmoothedHinge<float>;

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelSmoothedHingeDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelSmoothedHingeDouble)

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelSmoothedHingeFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelSmoothedHingeFloat)

#endif  // LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_SMOOTHED_HINGE_H_
