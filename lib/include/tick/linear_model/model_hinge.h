#ifndef LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_HINGE_H_
#define LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_HINGE_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"

#include <cereal/types/base_class.hpp>

template <class T>
class DLL_PUBLIC TModelHinge : public virtual TModelGeneralizedLinear<T> {
 protected:
  using TModelGeneralizedLinear<T>::compute_features_norm_sq;
  using TModelGeneralizedLinear<T>::n_samples;
  using TModelGeneralizedLinear<T>::features_norm_sq;
  using TModelGeneralizedLinear<T>::fit_intercept;

 public:
  using TModelGeneralizedLinear<T>::get_label;
  using TModelGeneralizedLinear<T>::use_intercept;
  using TModelGeneralizedLinear<T>::get_inner_prod;
  using TModelGeneralizedLinear<T>::get_class_name;

 public:
  // This exists soley for cereal/swig
  TModelHinge() : TModelHinge<T>(nullptr, nullptr, false) {}

  TModelHinge(const std::shared_ptr<BaseArray2d<T> > features,
              const std::shared_ptr<SArray<T> > labels,
              const bool fit_intercept, const int n_threads = 1)
      : TModelLabelsFeatures<T>(features, labels),
        TModelGeneralizedLinear<T>(features, labels, fit_intercept, n_threads) {
  }

  T loss_i(const ulong i, const Array<T> &coeffs) override;

  T grad_i_factor(const ulong i, const Array<T> &coeffs) override;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp(
        "ModelGeneralizedLinear",
        typename cereal::base_class<TModelGeneralizedLinear<T> >(this)));
  }

  BoolStrReport compare(const TModelHinge<T> &that, std::stringstream &ss) {
    ss << get_class_name() << std::endl;
    return TModelGeneralizedLinear<T>::compare(that, ss);
  }
  BoolStrReport compare(const TModelHinge<T> &that) {
    std::stringstream ss;
    return compare(that, ss);
  }
  BoolStrReport operator==(const TModelHinge<T> &that) {
    return TModelHinge<T>::compare(that);
  }
};

using ModelHinge = TModelHinge<double>;
using ModelHingeDouble = TModelHinge<double>;
using ModelHingeFloat = TModelHinge<float>;

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelHingeDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelHingeDouble)

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelHingeFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelHingeFloat)


#endif  // LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_HINGE_H_
