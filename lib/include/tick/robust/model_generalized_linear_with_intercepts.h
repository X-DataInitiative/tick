//
// Created by Stéphane GAIFFAS on 06/12/2015.
//
#ifndef LIB_INCLUDE_TICK_ROBUST_MODEL_GENERALIZED_LINEAR_WITH_INTERCEPTS_H_
#define LIB_INCLUDE_TICK_ROBUST_MODEL_GENERALIZED_LINEAR_WITH_INTERCEPTS_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"
#include "tick/base_model/model_labels_features.h"

template <class T, class K = T>
class DLL_PUBLIC TModelGeneralizedLinearWithIntercepts : public virtual TModelGeneralizedLinear<T, K> {
 private:
  std::string clazz = "TModelGeneralizedLinearWithIntercepts<"
    + std::string(typeid(T).name())
    + ", " + std::string(typeid(K).name()) + ">";

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

 protected:
  /**
  using TModelGeneralizedLinear<T, K>::n_samples;
   * Computes gradient fo ith observation
   * @param i : The selected observation
   * @param out : Preallocated vector in which information is store
   * @param coeffs : coefficient at which the gradient is computed
   * @param fill : If `true` out will be filled by the gradient value, otherwise out will be
   * inceremented by the gradient value.
   */
  virtual void compute_grad_i(const ulong i, const Array<T> &coeffs,
                      Array<K> &out, const bool fill);

 public:
  TModelGeneralizedLinearWithIntercepts(
    const std::shared_ptr<BaseArray2d<K> > features,
    const std::shared_ptr<SArray<K> > labels,
    const bool fit_intercept,
    const int n_threads = 1);

  virtual ~TModelGeneralizedLinearWithIntercepts() {}

  virtual const char *get_class_name() const {
    return clazz.c_str();
  }

  virtual void grad(const Array<T> &coeffs, Array<K> &out);

  virtual K loss(const Array<T> &coeffs);

  virtual K get_inner_prod(const ulong i, const Array<T> &coeffs) const;

  ulong get_n_coeffs() const override {
    return n_features + n_samples + static_cast<int>(fit_intercept);
  }
};

class DLL_PUBLIC ModelGeneralizedLinearWithIntercepts : public TModelGeneralizedLinearWithIntercepts<double, double> {
 public:
  ModelGeneralizedLinearWithIntercepts(const SBaseArrayDouble2dPtr features,
                                       const SArrayDoublePtr labels,
                                       const bool fit_intercept,
                                       const int n_threads = 1);
  const char *get_class_name() const override;
};

using TModelGeneralizedLinearWithInterceptsDouble = TModelGeneralizedLinearWithIntercepts<double, double>;

#endif  // LIB_INCLUDE_TICK_ROBUST_MODEL_GENERALIZED_LINEAR_WITH_INTERCEPTS_H_
