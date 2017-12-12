// License: BSD 3 clause

#include "tick/robust/model_epsilon_insensitive.h"

ModelEpsilonInsensitive::ModelEpsilonInsensitive(const SBaseArrayDouble2dPtr features,
                                                 const SArrayDoublePtr labels,
                                                 const bool fit_intercept,
                                                 const double threshold,
                                                 const int n_threads)

    : ModelGeneralizedLinear(features,
                             labels,
                             fit_intercept,
                             n_threads) {
  set_threshold(threshold);
}

const char *ModelEpsilonInsensitive::get_class_name() const {
  return "ModelEpsilonInsensitive";
}

double ModelEpsilonInsensitive::loss_i(const ulong i,
                                       const ArrayDouble &coeffs) {
  const double z = std::abs(get_inner_prod(i, coeffs) - get_label(i));
  if (z > threshold) {
    return z - threshold;
  } else {
    return 0.;
  }
}

double ModelEpsilonInsensitive::grad_i_factor(const ulong i,
                                              const ArrayDouble &coeffs) {
  const double d = get_inner_prod(i, coeffs) - get_label(i);
  if (std::abs(d) > threshold) {
    if (d > 0) {
      return 1;
    } else {
      return -1;
    }
  } else {
    return 0.;
  }
}
