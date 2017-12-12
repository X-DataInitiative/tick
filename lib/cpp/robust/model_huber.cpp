// License: BSD 3 clause

#include "tick/robust/model_huber.h"

ModelHuber::ModelHuber(const SBaseArrayDouble2dPtr features,
                       const SArrayDoublePtr labels,
                       const bool fit_intercept,
                       const double threshold,
                       const int n_threads)

    : ModelGeneralizedLinear(features,
                             labels,
                             fit_intercept,
                             n_threads),
      ModelLipschitz() {
  set_threshold(threshold);
}

const char *ModelHuber::get_class_name() const {
  return "ModelHuber";
}

double ModelHuber::loss_i(const ulong i,
                          const ArrayDouble &coeffs) {
  const double d = get_inner_prod(i, coeffs) - get_label(i);
  const double d_abs = std::abs(d);
  if (d_abs < threshold) {
    return d * d / 2;
  } else {
    return threshold * d_abs - threshold_squared_over_two;
  }
}

double ModelHuber::grad_i_factor(const ulong i,
                                 const ArrayDouble &coeffs) {
  const double d = get_inner_prod(i, coeffs) - get_label(i);
  if (std::abs(d) <= threshold) {
    return d;
  } else {
    if (d >= 0) {
      return threshold;
    } else {
      return -threshold;
    }
  }
}

void ModelHuber::compute_lip_consts() {
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
