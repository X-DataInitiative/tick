// License: BSD 3 clause

//
// Created by Stéphane GAIFFAS on 12/12/2015.
//

#include "tick/linear_model/model_linreg.h"

ModelLinReg::ModelLinReg(const SBaseArrayDouble2dPtr features,
                         const SArrayDoublePtr labels,
                         const bool fit_intercept,
                         const int n_threads)

    : ModelGeneralizedLinear(features,
                             labels,
                             fit_intercept,
                             n_threads),
      ModelLipschitz() {}

const char *ModelLinReg::get_class_name() const {
  return "ModelLinReg";
}

double ModelLinReg::sdca_dual_min_i(const ulong i,
                                    const double dual_i,
                                    const ArrayDouble &primal_vector,
                                    const double previous_delta_dual_i,
                                    double l_l2sq) {
  compute_features_norm_sq();
  double normalized_features_norm = features_norm_sq[i] / (l_l2sq * n_samples);
  if (use_intercept()) {
    normalized_features_norm += 1. / (l_l2sq * n_samples);
  }
  const double primal_dot_features = get_inner_prod(i, primal_vector);
  const double label = get_label(i);
  const double delta_dual = -(dual_i + primal_dot_features - label) / (1 + normalized_features_norm);
  return delta_dual;
}

double ModelLinReg::loss_i(const ulong i,
                           const ArrayDouble &coeffs) {
  // Compute x_i^T \beta + b
  const double z = get_inner_prod(i, coeffs);
  const double d = get_label(i) - z;
  return d * d / 2;
}

double ModelLinReg::grad_i_factor(const ulong i,
                                  const ArrayDouble &coeffs) {
  const double z = get_inner_prod(i, coeffs);
  return z - get_label(i);
}

void ModelLinReg::compute_lip_consts() {
  if (ready_lip_consts) {
    return;
  } else {
    compute_features_norm_sq();
    lip_consts = ArrayDouble(n_samples);
    for (ulong i = 0; i < n_samples; ++i) {
      if (fit_intercept) {
        lip_consts[i] = features_norm_sq[i] + 1;
      } else {
        lip_consts[i] = features_norm_sq[i];
      }
    }
  }
}
