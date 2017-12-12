// License: BSD 3 clause

#include "tick/robust/model_absolute_regression.h"

ModelAbsoluteRegression::ModelAbsoluteRegression(const SBaseArrayDouble2dPtr features,
                                                 const SArrayDoublePtr labels,
                                                 const bool fit_intercept,
                                                 const int n_threads)
    : ModelGeneralizedLinear(features,
                             labels,
                             fit_intercept,
                             n_threads) {}

const char *ModelAbsoluteRegression::get_class_name() const {
  return "ModelAbsoluteRegression";
}

double ModelAbsoluteRegression::loss_i(const ulong i,
                                       const ArrayDouble &coeffs) {
  return std::abs(get_inner_prod(i, coeffs) - get_label(i));
}

double ModelAbsoluteRegression::grad_i_factor(const ulong i,
                                              const ArrayDouble &coeffs) {
  const double d = get_inner_prod(i, coeffs) - get_label(i);
  if (d > 0) {
    return 1;
  } else {
    if (d < 0) {
      return -1;
    } else {
      return 0;
    }
  }
}
