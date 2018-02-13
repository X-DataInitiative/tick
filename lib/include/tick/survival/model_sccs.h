//
// Created by Maryan Morel on 18/05/2017.
//

#ifndef LIB_INCLUDE_TICK_SURVIVAL_MODEL_SCCS_H_
#define LIB_INCLUDE_TICK_SURVIVAL_MODEL_SCCS_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "tick/base_model/model_lipschitz.h"

template <class T>
class DLL_PUBLIC TModelSCCS : public TModelLipschitz<T> {
 protected:
  using SBaseArrayT2dPtrList1D = std::vector<std::shared_ptr<BaseArray2d<T> > >;
  using TModelLipschitz<T>::ready_lip_consts;
  using TModelLipschitz<T>::lip_consts;

 protected:
  ulong n_intervals;
  ulong n_lags;
  ulong n_samples;
  ulong n_observations;
  ulong n_lagged_features;
  ulong n_features;

  // Label vectors
  SArrayIntPtrList1D labels;

  // Feature matrices
  SBaseArrayT2dPtrList1D features;

  // Censoring vectors
  SBaseArrayULongPtr censoring;

 public:
  TModelSCCS(const SBaseArrayT2dPtrList1D &features,
             const SArrayIntPtrList1D &labels,
             const SBaseArrayULongPtr censoring, ulong n_lags);

  const char *get_class_name() const override {
    return "LongitudinalMultinomial";
  };

  T loss(const Array<T> &coeffs) override;

  T loss_i(const ulong i, const Array<T> &coeffs) override;

  void grad(const Array<T> &coeffs, Array<T> &out) override;

  void grad_i(const ulong i, const Array<T> &coeffs, Array<T> &out) override;

  void compute_lip_consts() override;

  ulong get_n_samples() const override { return n_samples; }

  ulong get_n_features() const override { return n_features; }

  ulong get_rand_max() { return n_samples; }

  ulong get_epoch_size() const override { return n_samples; }

  // Number of parameters to be estimated. Can differ from the number of
  // features, e.g. when using lags.
  ulong get_n_coeffs() const override { return n_lagged_features; }

  inline ulong get_max_interval(ulong i) const {
    return std::min(censoring->value(i), n_intervals);
  }

  bool is_sparse() const override { return false; }

  inline BaseArray<T> get_longitudinal_features(ulong i, ulong t) const {
    return view_row(*features[i], t);
  }

  inline T get_longitudinal_label(ulong i, ulong t) const {
    return view(*labels[i])[t];
  }

  T get_inner_prod(const ulong i, const ulong t, const Array<T> &coeffs) const;

  static inline T sumExpMinusMax(Array<T> &x, T x_max) {
    T sum = 0;
    for (ulong i = 0; i < x.size(); ++i)
      sum += exp(x[i] - x_max);  // overflow-proof
    return sum;
  }

  static inline T logSumExp(Array<T> &x) {
    T x_max = x.max();
    return x_max + log(sumExpMinusMax(x, x_max));
  }

  static inline void softMax(Array<T> &x, Array<T> &out) {
    T x_max = x.max();
    T sum = sumExpMinusMax(x, x_max);
    for (ulong i = 0; i < x.size(); i++) {
      out[i] = exp(x[i] - x_max) / sum;  // overflow-proof
    }
  }
};

using ModelSCCS = TModelSCCS<double>;

using ModelSCCSDouble = TModelSCCS<double>;

using ModelSCCSFloat = TModelSCCS<float>;

#endif  // LIB_INCLUDE_TICK_SURVIVAL_MODEL_SCCS_H_
