// License: BSD 3 clause

//
// Created by Martin Bompaire on 21/10/15.
//

#include "tick/linear_model/model_logreg.h"

ModelLogReg::ModelLogReg(const SBaseArrayDouble2dPtr features,
                         const SArrayDoublePtr labels,
                         const bool fit_intercept,
                         const int n_threads)
    : ModelGeneralizedLinear(features, labels, fit_intercept, n_threads),
      ModelLipschitz() {}

const char *ModelLogReg::get_class_name() const {
  return "ModelLogReg";
}

void ModelLogReg::sigmoid(const ArrayDouble &x, ArrayDouble &out) {
  for (ulong i = 0; i < x.size(); ++i) {
    out[i] = sigmoid(x[i]);
  }
}

void ModelLogReg::logistic(const ArrayDouble &x, ArrayDouble &out) {
  for (ulong i = 0; i < x.size(); ++i) {
    out[i] = logistic(x[i]);
  }
}

double ModelLogReg::loss_i(const ulong i, const ArrayDouble &coeffs) {
  double z_i = get_inner_prod(i, coeffs);
  z_i *= get_label(i);
  return logistic(z_i);
}

double ModelLogReg::grad_i_factor(const ulong i, const ArrayDouble &coeffs) {
  // The label in { -1, 1 }
  const double y_i = get_label(i);
  // Contains x_i^T w + b
  const double z_i = get_inner_prod(i, coeffs);

  return y_i * (sigmoid(y_i * z_i) - 1);
}

double ModelLogReg::sdca_dual_min_i(const ulong i,
                                    const double dual_i,
                                    const ArrayDouble &primal_vector,
                                    const double previous_delta_dual_i,
                                    double l_l2sq) {
  compute_features_norm_sq();
  double epsilon = 1e-1;
  double normalized_features_norm = features_norm_sq[i] / (l_l2sq * n_samples);
  if (use_intercept()) {
    normalized_features_norm += 1. / (l_l2sq * n_samples);
  }
  const double primal_dot_features = get_inner_prod(i, primal_vector);
  const double label = get_label(i);
  double new_dual_times_label{0.};

  // initial delta dual as suggested in original paper
  // http://www.jmlr.org/papers/volume14/shalev-shwartz13a/shalev-shwartz13a.pdf 6.2
  double delta_dual = label / (1. + exp(primal_dot_features * label)) - dual_i;
  delta_dual /= std::max(1., 0.25 + normalized_features_norm);

  for (int j = 0; j < 10; ++j) {
    double new_dual = dual_i + delta_dual;
    new_dual_times_label = new_dual * label;
    // Check we are in the correct bounds
    if (new_dual_times_label <= 0) {
      new_dual = epsilon / label;
      delta_dual = new_dual - dual_i;
      new_dual_times_label = new_dual * label;
      epsilon *= 1e-1;
    }
    if (new_dual_times_label >= 1) {
      new_dual = (1 - epsilon) / label;
      delta_dual = new_dual - dual_i;
      new_dual_times_label = new_dual * label;
      epsilon *= 1e-1;
    }

    // Do newton descent
    // Logistic loss part
    double f_prime = label * (log(new_dual_times_label) - log(1 - new_dual_times_label));
    double f_second = 1 / (new_dual_times_label * (1 - new_dual_times_label));

    // Ridge regression part
    f_prime += normalized_features_norm * delta_dual + primal_dot_features;
    f_second += normalized_features_norm;

    delta_dual -= f_prime / f_second;
    new_dual = dual_i + delta_dual;
    new_dual_times_label = new_dual * label;

    if (std::abs(f_prime / f_second) < 1e-10) {
      break;
    }
  }
  // Check we are in the correct bounds
  if (new_dual_times_label <= 0) {
    double new_dual = epsilon / label;
    delta_dual = new_dual - dual_i;
    new_dual_times_label = new_dual * label;
  }
  if (new_dual_times_label >= 1) {
    double new_dual = (1 - epsilon) / label;
    delta_dual = new_dual - dual_i;
  }

  return delta_dual;
}

void ModelLogReg::compute_lip_consts() {
  if (ready_lip_consts) {
    return;
  } else {
    compute_features_norm_sq();
    lip_consts = ArrayDouble(n_samples);
    for (ulong i = 0; i < n_samples; ++i) {
      if (fit_intercept) {
        lip_consts[i] = (features_norm_sq[i] + 1) / 4;
      } else {
        lip_consts[i] = features_norm_sq[i] / 4;
      }
    }
  }
}
