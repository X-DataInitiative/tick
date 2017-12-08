// License: BSD 3 clause

#include "tick/robust/model_modified_huber.h"

ModelModifiedHuber::ModelModifiedHuber(const SBaseArrayDouble2dPtr features,
                                       const SArrayDoublePtr labels,
                                       const bool fit_intercept,
                                       const int n_threads)

    : ModelGeneralizedLinear(features,
                             labels,
                             fit_intercept,
                             n_threads),
      ModelLipschitz() {}

const char *ModelModifiedHuber::get_class_name() const {
  return "ModelModifiedHuber";
}

double ModelModifiedHuber::loss_i(const ulong i,
                                  const ArrayDouble &coeffs) {
  const double z = get_label(i) * get_inner_prod(i, coeffs);
  if (z >= 1) {
    return 0.;
  } else {
    if (z <= -1) {
      return -4 * z;
    } else {
      const double d = 1 - z;
      return d * d;
    }
  }
}

double ModelModifiedHuber::grad_i_factor(const ulong i,
                                         const ArrayDouble &coeffs) {
  const double y = get_label(i);
  const double z = y * get_inner_prod(i, coeffs);
  if (z >= 1) {
    return 0.;
  } else {
    if (z <= -1) {
      return -4 * y;
    } else {
      return 2 * y * (z - 1);
    }
  }
}

void ModelModifiedHuber::compute_lip_consts() {
  if (ready_lip_consts) {
    return;
  } else {
    compute_features_norm_sq();
    lip_consts = ArrayDouble(n_samples);
    for (ulong i = 0; i < n_samples; ++i) {
      if (fit_intercept) {
        lip_consts[i] = 2 * (features_norm_sq[i] + 1);
      } else {
        lip_consts[i] = 2 * features_norm_sq[i];
      }
    }
  }
}
