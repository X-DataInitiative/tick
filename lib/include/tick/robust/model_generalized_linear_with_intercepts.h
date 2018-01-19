//
// Created by Stéphane GAIFFAS on 06/12/2015.
//
#ifndef LIB_INCLUDE_TICK_ROBUST_MODEL_GENERALIZED_LINEAR_WITH_INTERCEPTS_H_
#define LIB_INCLUDE_TICK_ROBUST_MODEL_GENERALIZED_LINEAR_WITH_INTERCEPTS_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"
#include "tick/base_model/model_labels_features.h"

class DLL_PUBLIC ModelGeneralizedLinearWithIntercepts : public virtual ModelGeneralizedLinear {
 protected:
  /**
   * Computes gradient fo ith observation
   * @param i : The selected observation
   * @param out : Preallocated vector in which information is store
   * @param coeffs : coefficient at which the gradient is computed
   * @param fill : If `true` out will be filled by the gradient value, otherwise out will be
   * inceremented by the gradient value.
   */
  void compute_grad_i(const ulong i, const ArrayDouble &coeffs,
                      ArrayDouble &out, const bool fill) override;

 public:
  ModelGeneralizedLinearWithIntercepts(const SBaseArrayDouble2dPtr features,
                                       const SArrayDoublePtr labels,
                                       const bool fit_intercept,
                                       const int n_threads = 1);

  const char *get_class_name() const override;

  void grad(const ArrayDouble &coeffs, ArrayDouble &out) override;

  double loss(const ArrayDouble &coeffs) override;

  double get_inner_prod(const ulong i, const ArrayDouble &coeffs) const override;

  ulong get_n_coeffs() const override {
    return n_features + n_samples + static_cast<int>(fit_intercept);
  }
};

#endif  // LIB_INCLUDE_TICK_ROBUST_MODEL_GENERALIZED_LINEAR_WITH_INTERCEPTS_H_
