
#ifndef LIB_INCLUDE_TICK_ROBUST_MODEL_EPSILON_INSENSITIVE_H_
#define LIB_INCLUDE_TICK_ROBUST_MODEL_EPSILON_INSENSITIVE_H_

// License: BSD 3 clause

#include <cereal/types/base_class.hpp>
#include "tick/base_model/model_generalized_linear.h"

template <class T>
class DLL_PUBLIC TModelEpsilonInsensitive
    : public virtual TModelGeneralizedLinear<T> {
  // Grants cereal access to default constructor
  friend class cereal::access;

 protected:
  using TModelGeneralizedLinear<T>::features_norm_sq;
  using TModelGeneralizedLinear<T>::compute_features_norm_sq;
  using TModelGeneralizedLinear<T>::n_samples;
  using TModelGeneralizedLinear<T>::n_features;
  using TModelGeneralizedLinear<T>::fit_intercept;
  using TModelGeneralizedLinear<T>::compute_grad_i;
  using TModelGeneralizedLinear<T>::n_threads;
  using TModelGeneralizedLinear<T>::get_inner_prod;

 public:
  using TModelGeneralizedLinear<T>::get_label;
  using TModelGeneralizedLinear<T>::grad_i;
  using TModelGeneralizedLinear<T>::get_features;
  using TModelGeneralizedLinear<T>::grad_i_factor;
  using TModelGeneralizedLinear<T>::get_class_name;

 private:
  T threshold;

 private:
  // This exists soley for cereal which has friend access
  TModelEpsilonInsensitive()
      : TModelEpsilonInsensitive(nullptr, nullptr, false, 1) {}

 public:
  TModelEpsilonInsensitive(const std::shared_ptr<BaseArray2d<T>> features,
                           const std::shared_ptr<SArray<T>> labels,
                           const bool fit_intercept, const T threshold,
                           const int n_threads = 1);

  T loss_i(const ulong i, const Array<T> &coeffs) override;

  T grad_i_factor(const ulong i, const Array<T> &coeffs) override;

  virtual T get_threshold(void) const { return threshold; }

  virtual void set_threshold(const T threshold) {
    if (threshold <= 0.) {
      TICK_ERROR("threshold must be > 0");
    } else {
      this->threshold = threshold;
    }
  }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("ModelGeneralizedLinear",
                        cereal::base_class<TModelGeneralizedLinear<T>>(this)));
    ar(threshold);
  }

  BoolStrReport compare(const TModelEpsilonInsensitive<T> &that,
                        std::stringstream &ss) {
    ss << get_class_name() << std::endl;
    auto are_equal = TModelGeneralizedLinear<T>::compare(that, ss) &&
                     TICK_CMP_REPORT(ss, threshold);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport compare(const TModelEpsilonInsensitive<T> &that) {
    std::stringstream ss;
    return compare(that, ss);
  }
  BoolStrReport operator==(const TModelEpsilonInsensitive<T> &that) {
    return TModelEpsilonInsensitive<T>::compare(that);
  }
};

using ModelEpsilonInsensitive = TModelEpsilonInsensitive<double>;

using ModelEpsilonInsensitiveDouble = TModelEpsilonInsensitive<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelEpsilonInsensitiveDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelEpsilonInsensitiveDouble)

using ModelEpsilonInsensitiveFloat = TModelEpsilonInsensitive<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelEpsilonInsensitiveFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelEpsilonInsensitiveFloat)

#endif  // LIB_INCLUDE_TICK_ROBUST_MODEL_EPSILON_INSENSITIVE_H_
