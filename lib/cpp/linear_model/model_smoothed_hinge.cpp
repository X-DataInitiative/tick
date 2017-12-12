// License: BSD 3 clause

#include "tick/linear_model/model_smoothed_hinge.h"

ModelSmoothedHinge::ModelSmoothedHinge(const SBaseArrayDouble2dPtr features,
                                       const SArrayDoublePtr labels,
                                       const bool fit_intercept,
                                       const double smoothness,
                                       const int n_threads)
    : ModelGeneralizedLinear(features,
                             labels,
                             fit_intercept,
                             n_threads),
      ModelLipschitz() {
  set_smoothness(smoothness);
}

const char *ModelSmoothedHinge::get_class_name() const {
  return "ModelSmoothedHinge";
}

double ModelSmoothedHinge::loss_i(const ulong i,
                                  const ArrayDouble &coeffs) {
  const double z = get_label(i) * get_inner_prod(i, coeffs);
  if (z >= 1) {
    return 0.;
  } else {
    if (z <= 1 - smoothness) {
      return 1 - z - smoothness / 2;
    } else {
      const double d = (1 - z);
      return d * d / (2 * smoothness);
    }
  }
}

double ModelSmoothedHinge::grad_i_factor(const ulong i,
                                         const ArrayDouble &coeffs) {
  const double y = get_label(i);
  const double z = y * get_inner_prod(i, coeffs);
  if (z >= 1) {
    return 0.;
  } else {
    if (z <= 1 - smoothness) {
      return -y;
    } else {
      return (z - 1) * y / smoothness;
    }
  }
}

void ModelSmoothedHinge::compute_lip_consts() {
  if (ready_lip_consts) {
    return;
  } else {
    compute_features_norm_sq();
    lip_consts = ArrayDouble(n_samples);
    for (ulong i = 0; i < n_samples; ++i) {
      if (fit_intercept) {
        lip_consts[i] = (features_norm_sq[i] + 1) / smoothness;
      } else {
        lip_consts[i] = features_norm_sq[i] / smoothness;
      }
    }
  }
}
