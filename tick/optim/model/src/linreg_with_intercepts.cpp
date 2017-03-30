//
// Created by Stéphane GAIFFAS on 12/12/2015.
//

#include "linreg_with_intercepts.h"

ModelLinRegWithIntercepts::ModelLinRegWithIntercepts(const SBaseArrayDouble2dPtr features,
                                                     const SArrayDoublePtr labels,
                                                     const int n_threads)
    : ModelGeneralizedLinearWithIntercepts(features, labels, n_threads),
      ModelLipschitz() {}

const char *ModelLinRegWithIntercepts::get_class_name() const {
  return "ModelLinRegWithIntercepts";
}

double ModelLinRegWithIntercepts::loss_i(const ulong i, const ArrayDouble &coeffs) {
  // Compute x_i^T \beta + b_i
  const double z = get_inner_prod(i, coeffs);
  const double d = get_label(i) - z;
  return d * d / 2;
}

double ModelLinRegWithIntercepts::grad_i_factor(const ulong i, const ArrayDouble &coeffs) {
  const double z = get_inner_prod(i, coeffs);
  return z - get_label(i);
}

void ModelLinRegWithIntercepts::compute_lip_consts() {
  if (ready_lip_consts) {
    return;
  } else {
    compute_features_norm_sq();
    lip_consts = ArrayDouble(n_samples);
    for (ulong i = 0; i < n_samples; ++i) {
      lip_consts[i] = features_norm_sq[i] + 1;
    }
  }
}
