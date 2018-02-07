//
// Created by Stéphane GAIFFAS on 06/12/2015.
//
#ifndef LIB_INCLUDE_TICK_ROBUST_MODEL_GENERALIZED_LINEAR_WITH_INTERCEPTS_H_
#define LIB_INCLUDE_TICK_ROBUST_MODEL_GENERALIZED_LINEAR_WITH_INTERCEPTS_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"
#include "tick/base_model/model_labels_features.h"

template <class T>
class DLL_PUBLIC TModelGeneralizedLinearWithIntercepts
    : public virtual TModelGeneralizedLinear<T> {
 private:
  std::string clazz = "TModelGeneralizedLinearWithIntercepts<" +
                      std::string(typeid(T).name()) + ">";

 protected:
  using TModelGeneralizedLinear<T>::features_norm_sq;
  using TModelGeneralizedLinear<T>::compute_features_norm_sq;
  using TModelGeneralizedLinear<T>::n_samples;
  using TModelGeneralizedLinear<T>::n_features;
  using TModelGeneralizedLinear<T>::fit_intercept;
  using TModelGeneralizedLinear<T>::compute_grad_i;
  using TModelGeneralizedLinear<T>::n_threads;

 public:
  using TModelGeneralizedLinear<T>::grad_i;
  using TModelGeneralizedLinear<T>::get_features;
  using TModelGeneralizedLinear<T>::grad_i_factor;

 protected:
  /**
   * Computes gradient fo ith observation
   * @param i : The selected observation
   * @param out : Preallocated vector in which information is store
   * @param coeffs : coefficient at which the gradient is computed
   * @param fill : If `true` out will be filled by the gradient value, otherwise
   * out will be inceremented by the gradient value.
   */
  virtual void compute_grad_i(const ulong i, const Array<T> &coeffs,
                              Array<T> &out, const bool fill);

 public:
  TModelGeneralizedLinearWithIntercepts(
      const std::shared_ptr<BaseArray2d<T> > features,
      const std::shared_ptr<SArray<T> > labels, const bool fit_intercept,
      const int n_threads = 1);

  virtual ~TModelGeneralizedLinearWithIntercepts() {}

  virtual const char *get_class_name() const { return clazz.c_str(); }

  virtual void grad(const Array<T> &coeffs, Array<T> &out);

  virtual T loss(const Array<T> &coeffs);

  virtual T get_inner_prod(const ulong i, const Array<T> &coeffs) const;

  ulong get_n_coeffs() const override {
    return n_features + n_samples + static_cast<int>(fit_intercept);
  }
};

using ModelGeneralizedLinearWithIntercepts =
    TModelGeneralizedLinearWithIntercepts<double>;
using ModelGeneralizedLinearWithInterceptsDouble =
    TModelGeneralizedLinearWithIntercepts<double>;
using ModelGeneralizedLinearWithInterceptsFloat =
    TModelGeneralizedLinearWithIntercepts<float>;

#endif  // LIB_INCLUDE_TICK_ROBUST_MODEL_GENERALIZED_LINEAR_WITH_INTERCEPTS_H_
