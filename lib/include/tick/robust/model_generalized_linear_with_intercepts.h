//
// Created by Stéphane GAIFFAS on 06/12/2015.
//
#ifndef LIB_INCLUDE_TICK_ROBUST_MODEL_GENERALIZED_LINEAR_WITH_INTERCEPTS_H_
#define LIB_INCLUDE_TICK_ROBUST_MODEL_GENERALIZED_LINEAR_WITH_INTERCEPTS_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"
#include "tick/base_model/model_labels_features.h"

template <class T, class K = T>
class DLL_PUBLIC TModelGeneralizedLinearWithIntercepts
    : public virtual TModelGeneralizedLinear<T, K> {
 protected:
  using TModelGeneralizedLinear<T, K>::features_norm_sq;
  using TModelGeneralizedLinear<T, K>::compute_features_norm_sq;
  using TModelGeneralizedLinear<T, K>::n_samples;
  using TModelGeneralizedLinear<T, K>::n_features;
  using TModelGeneralizedLinear<T, K>::fit_intercept;
  using TModelGeneralizedLinear<T, K>::compute_grad_i;
  using TModelGeneralizedLinear<T, K>::n_threads;

 public:
  using TModelGeneralizedLinear<T, K>::grad_i;
  using TModelGeneralizedLinear<T, K>::get_features;
  using TModelGeneralizedLinear<T, K>::grad_i_factor;
  using TModelGeneralizedLinear<T, K>::get_class_name;

 protected:
  /**
   * Computes gradient fo ith observation
   * @param i : The selected observation
   * @param out : Preallocated vector in which information is store
   * @param coeffs : coefficient at which the gradient is computed
   * @param fill : If `true` out will be filled by the gradient value, otherwise
   * out will be inceremented by the gradient value.
   */
  void compute_grad_i(const ulong i, const Array<K> &coeffs, Array<T> &out,
                      const bool fill) override;

 public:
  // This exists soley for cereal/swig
  TModelGeneralizedLinearWithIntercepts()
      : TModelGeneralizedLinearWithIntercepts(nullptr, nullptr, false) {}

  TModelGeneralizedLinearWithIntercepts(
      const std::shared_ptr<BaseArray2d<T>> features,
      const std::shared_ptr<SArray<T>> labels, const bool fit_intercept,
      const int n_threads = 1)
      : TModelLabelsFeatures<T, K>(features, labels),
        TModelGeneralizedLinear<T, K>(features, labels, fit_intercept,
                                      n_threads) {}

  virtual ~TModelGeneralizedLinearWithIntercepts() {}

  void grad(const Array<K> &coeffs, Array<T> &out) override;

  T loss(const Array<K> &coeffs) override;

  T get_inner_prod(const ulong i, const Array<K> &coeffs) const override;

  ulong get_n_coeffs() const override {
    return n_features + n_samples + static_cast<int>(fit_intercept);
  }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp(
        "ModelGeneralizedLinear",
        cereal::virtual_base_class<TModelGeneralizedLinear<T, K>>(this)));
  }

  BoolStrReport compare(const TModelGeneralizedLinearWithIntercepts<T, K> &that,
                        std::stringstream &ss) {
    ss << get_class_name() << std::endl;
    return TModelGeneralizedLinear<T, K>::compare(that, ss);
  }
  BoolStrReport compare(
      const TModelGeneralizedLinearWithIntercepts<T, K> &that) {
    std::stringstream ss;
    return compare(that, ss);
  }
  BoolStrReport operator==(
      const TModelGeneralizedLinearWithIntercepts<T, K> &that) {
    return TModelGeneralizedLinearWithIntercepts<T, K>::compare(that);
  }
};

using ModelGeneralizedLinearWithIntercepts =
    TModelGeneralizedLinearWithIntercepts<double>;

using ModelGeneralizedLinearWithInterceptsDouble =
    TModelGeneralizedLinearWithIntercepts<double>;

using ModelGeneralizedLinearWithInterceptsFloat =
    TModelGeneralizedLinearWithIntercepts<float>;

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelGeneralizedLinearWithInterceptsDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelGeneralizedLinearWithInterceptsDouble)

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelGeneralizedLinearWithInterceptsFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelGeneralizedLinearWithInterceptsFloat)

#endif  // LIB_INCLUDE_TICK_ROBUST_MODEL_GENERALIZED_LINEAR_WITH_INTERCEPTS_H_
