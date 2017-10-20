// License: BSD 3 clause

//
// Created by Stéphane GAIFFAS on 12/12/2015.
//

#include "linreg.h"

ModelLinReg::ModelLinReg(const SBaseArrayDouble2dPtr features,
                         const SArrayDoublePtr labels,
                         const bool fit_intercept,
                         const int n_threads)

    : ModelGeneralizedLinear(features,
                             labels,
                             fit_intercept,
                             n_threads),
      ModelLipschitz() {
  out = ArrayDouble(get_n_samples());
}

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

double ModelLinReg::loss2(const ArrayDouble &coeffs) {
//  out = *labels;
//  features->dot_incr(coeffs, -1., out);
//  return 0.5 * out.norm_sq() / n_samples;
  return parallel_map_additive_reduce(n_threads, n_threads, &ModelLinReg::loss2_split_i,
                                      this, coeffs);
}

double ModelLinReg::loss2_split_i(ulong i, const ArrayDouble &coeffs) {
  const ulong start = i == 0? 0 : i * n_samples / n_threads;
  const ulong end = i == n_threads - 1? n_samples - 1: (i + 1) * n_samples / n_threads;
  BaseArrayDouble2d features_rows = view_rows(*features, start, end);
  ArrayDouble out_rows = view(out, start, end);
  ArrayDouble labels_rows = view(*labels, start, end);
  out_rows.mult_fill(labels_rows, 1.);
  features_rows.dot_incr(coeffs, -1., out_rows);
  return 0.5 * out_rows.norm_sq() / n_samples;
}

void ModelLinReg::grad2(const ArrayDouble &coeffs, ArrayDouble &out) {
  features->dot_incr(coeffs, -1., out);
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
