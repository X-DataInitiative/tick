#ifndef LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_QUADRATIC_HINGE_H_
#define LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_QUADRATIC_HINGE_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"
#include "tick/base_model/model_lipschitz.h"

#include <cereal/types/base_class.hpp>

template <class T>
class DLL_PUBLIC TModelQuadraticHinge
    : public virtual TModelGeneralizedLinear<T>,
      public TModelLipschitz<T> {
  // Grants cereal access to default constructor
  friend class cereal::access;

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
  // This exists soley for cereal which has friend access
  TModelQuadraticHinge() : TModelQuadraticHinge<T>(nullptr, nullptr, false) {}

 public:
  TModelQuadraticHinge(const std::shared_ptr<BaseArray2d<T>> features,
                       const std::shared_ptr<SArray<T>> labels,
                       const bool fit_intercept, const int n_threads = 1)
      : TModelLabelsFeatures<T>(features, labels),
        TModelGeneralizedLinear<T>(features, labels, fit_intercept, n_threads) {
  }

  T loss_i(const ulong i, const Array<T> &coeffs) override;

  T grad_i_factor(const ulong i, const Array<T> &coeffs) override;

  void compute_lip_consts() override;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp(
        "ModelGeneralizedLinear",
        typename cereal::base_class<TModelGeneralizedLinear<T>>(this)));
    ar(cereal::make_nvp("ModelLipschitz",
                        typename cereal::base_class<TModelLipschitz<T>>(this)));
  }

  BoolStrReport compare(const TModelQuadraticHinge<T> &that,
                        std::stringstream &ss) {
    ss << get_class_name() << std::endl;
    bool are_equal = TModelGeneralizedLinear<T>::compare(that, ss) &&
                     TModelLipschitz<T>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport compare(const TModelQuadraticHinge<T> &that) {
    std::stringstream ss;
    return compare(that, ss);
  }
  BoolStrReport operator==(const TModelQuadraticHinge<T> &that) {
    return TModelQuadraticHinge<T>::compare(that);
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

#endif  // LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_QUADRATIC_HINGE_H_
