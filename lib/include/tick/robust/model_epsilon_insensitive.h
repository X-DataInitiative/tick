
#ifndef LIB_INCLUDE_TICK_ROBUST_MODEL_EPSILON_INSENSITIVE_H_
#define LIB_INCLUDE_TICK_ROBUST_MODEL_EPSILON_INSENSITIVE_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"

template <class T, class K = T>
class DLL_PUBLIC TModelEpsilonInsensitive
    : public virtual TModelGeneralizedLinear<T, K> {
 protected:
  using TModelGeneralizedLinear<T, K>::features_norm_sq;
  using TModelGeneralizedLinear<T, K>::compute_features_norm_sq;
  using TModelGeneralizedLinear<T, K>::n_samples;
  using TModelGeneralizedLinear<T, K>::n_features;
  using TModelGeneralizedLinear<T, K>::fit_intercept;
  using TModelGeneralizedLinear<T, K>::compute_grad_i;
  using TModelGeneralizedLinear<T, K>::n_threads;
  using TModelGeneralizedLinear<T, K>::get_inner_prod;

 public:
  using TModelGeneralizedLinear<T, K>::get_label;
  using TModelGeneralizedLinear<T, K>::grad_i;
  using TModelGeneralizedLinear<T, K>::get_features;
  using TModelGeneralizedLinear<T, K>::grad_i_factor;
  using TModelGeneralizedLinear<T, K>::get_class_name;

 private:
  T threshold;

 public:
  // This exists soley for cereal/swig
  TModelEpsilonInsensitive()
      : TModelEpsilonInsensitive(nullptr, nullptr, false, 1) {}

  TModelEpsilonInsensitive(const std::shared_ptr<BaseArray2d<T>> features,
                           const std::shared_ptr<SArray<T>> labels,
                           const bool fit_intercept, const T threshold,
                           const int n_threads = 1)
      : TModelLabelsFeatures<T, K>(features, labels),
        TModelGeneralizedLinear<T, K>(features, labels, fit_intercept,
                                      n_threads) {
    set_threshold(threshold);
  }

  T loss_i(const ulong i, const Array<K> &coeffs) override;

  T grad_i_factor(const ulong i, const Array<K> &coeffs) override;

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
    ar(cereal::make_nvp(
        "ModelGeneralizedLinear",
        cereal::virtual_base_class<TModelGeneralizedLinear<T, K>>(this)));
    ar(threshold);
  }

  BoolStrReport compare(const TModelEpsilonInsensitive<T, K> &that,
                        std::stringstream &ss) {
    ss << get_class_name() << std::endl;
    auto are_equal = TModelGeneralizedLinear<T, K>::compare(that, ss) &&
                     TICK_CMP_REPORT(ss, threshold);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport compare(const TModelEpsilonInsensitive<T, K> &that) {
    std::stringstream ss;
    return compare(that, ss);
  }
  BoolStrReport operator==(const TModelEpsilonInsensitive<T, K> &that) {
    return TModelEpsilonInsensitive<T, K>::compare(that);
  }
};

using ModelEpsilonInsensitive = TModelEpsilonInsensitive<double>;
using ModelEpsilonInsensitiveDouble = TModelEpsilonInsensitive<double>;
using ModelEpsilonInsensitiveFloat = TModelEpsilonInsensitive<float>;

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelEpsilonInsensitiveDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelEpsilonInsensitiveDouble)

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelEpsilonInsensitiveFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelEpsilonInsensitiveFloat)

#endif  // LIB_INCLUDE_TICK_ROBUST_MODEL_EPSILON_INSENSITIVE_H_
