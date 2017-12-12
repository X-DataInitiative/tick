// License: BSD 3 clause

#include "tick/linear_model/model_hinge.h"

ModelHinge::ModelHinge(const SBaseArrayDouble2dPtr features,
                       const SArrayDoublePtr labels,
                       const bool fit_intercept,
                       const int n_threads)

    : ModelGeneralizedLinear(features,
                             labels,
                             fit_intercept,
                             n_threads) {}

const char *ModelHinge::get_class_name() const {
  return "ModelHinge";
}

double ModelHinge::loss_i(const ulong i,
                          const ArrayDouble &coeffs) {
  const double z = get_label(i) * get_inner_prod(i, coeffs);
  if (z <= 1.) {
    return 1 - z;
  } else {
    return 0.;
  }
}

double ModelHinge::grad_i_factor(const ulong i,
                                 const ArrayDouble &coeffs) {
  const double y = get_label(i);
  const double z = y * get_inner_prod(i, coeffs);
  if (z <= 1.) {
    return -y;
  } else {
    return 0;
  }
}
