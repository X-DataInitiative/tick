// License: BSD 3 clause

#include "tick/robust/model_linreg_with_intercepts.h"

ModelLinRegWithIntercepts::ModelLinRegWithIntercepts(const SBaseArrayDouble2dPtr features,
                                                     const SArrayDoublePtr labels,
                                                     const bool fit_intercept,
                                                     const int n_threads)
    : TModelLabelsFeatures<double, double>(features, labels),
      TModelGeneralizedLinear<double, double>(features, labels, fit_intercept, n_threads) ,
      ModelGeneralizedLinearWithIntercepts(features, labels, fit_intercept, n_threads),
      ModelLinReg(features, labels, fit_intercept, n_threads) {}

const char *ModelLinRegWithIntercepts::get_class_name() const {
  return "ModelLinRegWithIntercepts";
}

void ModelLinRegWithIntercepts::compute_lip_consts() {
  if (ready_lip_consts) {
    return;
  } else {
    compute_features_norm_sq();
    lip_consts = ArrayDouble(get_n_samples());
    double c = 1;
    if (use_intercept()) {
      c = 2;
    }
    for (ulong i = 0; i < get_n_samples(); ++i) {
      lip_consts[i] = get_features_norm_sq()[i] + c;
    }
  }
}
