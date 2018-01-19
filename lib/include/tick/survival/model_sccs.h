//
// Created by Maryan Morel on 18/05/2017.
//

#ifndef LIB_INCLUDE_TICK_SURVIVAL_MODEL_SCCS_H_
#define LIB_INCLUDE_TICK_SURVIVAL_MODEL_SCCS_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "tick/base_model/model_lipschitz.h"

class DLL_PUBLIC ModelSCCS : public ModelLipschitz  {
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
  SBaseArrayDouble2dPtrList1D features;

  // Censoring vectors
  SBaseArrayULongPtr censoring;

 public:
  ModelSCCS(const SBaseArrayDouble2dPtrList1D &features,
                          const SArrayIntPtrList1D &labels,
                          const SBaseArrayULongPtr censoring,
                          ulong n_lags);

  const char *get_class_name() const override {
    return "LongitudinalMultinomial";
  };

  double loss(const ArrayDouble &coeffs) override;

  double loss_i(const ulong i, const ArrayDouble &coeffs) override;

  void grad(const ArrayDouble &coeffs, ArrayDouble &out) override;

  void grad_i(const ulong i,
              const ArrayDouble &coeffs,
              ArrayDouble &out) override;

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

  inline BaseArrayDouble get_longitudinal_features(ulong i,
                                                   ulong t) const {
    return view_row(*features[i], t);
  }

  inline double get_longitudinal_label(ulong i, ulong t) const {
    return view(*labels[i])[t];
  }

  double get_inner_prod(const ulong i,
                        const ulong t,
                        const ArrayDouble &coeffs) const;

  static inline double sumExpMinusMax(ArrayDouble &x, double x_max) {
    double sum = 0;
    for (ulong i = 0; i < x.size(); ++i) sum += exp(x[i] - x_max);  // overflow-proof
    return sum;
  }

  static inline double logSumExp(ArrayDouble &x) {
    double x_max = x.max();
    return x_max + log(sumExpMinusMax(x, x_max));
  }

  static inline void softMax(ArrayDouble &x, ArrayDouble &out) {
    double x_max = x.max();
    double sum = sumExpMinusMax(x, x_max);
    for (ulong i = 0; i < x.size(); i++) {
      out[i] = exp(x[i] - x_max) / sum;  // overflow-proof
    }
  }
};

#endif  // LIB_INCLUDE_TICK_SURVIVAL_MODEL_SCCS_H_
