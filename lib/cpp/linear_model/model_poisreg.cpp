// License: BSD 3 clause

//
// Created by Martin Bompaire on 21/10/15.
//

#include "tick/linear_model/model_poisreg.h"

ModelPoisReg::ModelPoisReg(const SBaseArrayDouble2dPtr features,
                           const SArrayDoublePtr labels,
                           const LinkType link_type,
                           const bool fit_intercept,
                           const int n_threads)
  : ModelGeneralizedLinear(features,
                           labels,
                           fit_intercept,
                           n_threads),
    link_type(link_type), ready_non_zero_label_map(false) {}

double ModelPoisReg::sdca_dual_min_i(const ulong i,
                                     const double dual_i,
                                     const ArrayDouble &primal_vector,
                                     const double previous_delta_dual_i,
                                     double l_l2sq) {
  if (link_type == LinkType::identity) {
    return sdca_dual_min_i_identity(i, dual_i, primal_vector, previous_delta_dual_i, l_l2sq);
  } else {
    return sdca_dual_min_i_exponential(i, dual_i, primal_vector, previous_delta_dual_i, l_l2sq);
  }
}

double ModelPoisReg::sdca_dual_min_i_exponential(const ulong i,
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
  double delta_dual = previous_delta_dual_i;
  const double label = get_label(i);
  double new_dual{0.};

  for (int j = 0; j < 10; ++j) {
    new_dual = dual_i + delta_dual;

    // Check we are in the correct bounds
    if (new_dual >= label) {
      new_dual = label - epsilon;
      delta_dual = new_dual - dual_i;
      epsilon *= 1e-1;
    }

    // Do newton descent
    // Poisson loss part
    double f_prime = -log(label - new_dual);
    double f_second = 1. / (label - new_dual);

    // Ridge regression part
    f_prime += normalized_features_norm * delta_dual + primal_dot_features;
    f_second += normalized_features_norm;

    delta_dual -= f_prime / f_second;
    new_dual = dual_i + delta_dual;

    if (std::abs(f_prime / f_second) < 1e-10) {
      break;
    }
  }
  // Check we are in the correct bounds
  if (new_dual >= label) {
    new_dual = label - epsilon;
    delta_dual = new_dual - dual_i;
  }

  return delta_dual;
}

void ModelPoisReg::init_non_zero_label_map() {
  non_zero_labels = VArrayULong::new_ptr();
  for (ulong i = 0; i < get_n_samples(); ++i) {
    if (get_label(i) != 0) {
      non_zero_labels->append1(i);
    }
  }
  n_non_zeros_labels = non_zero_labels->size();
  ready_non_zero_label_map = true;
}

double ModelPoisReg::sdca_dual_min_i_identity(const ulong i,
                                              const double dual_i,
                                              const ArrayDouble &primal_vector,
                                              const double previous_delta_dual_i,
                                              double l_l2sq) {
  if (!ready_features_norm_sq) {
    compute_features_norm_sq();
  }

  const double label = get_label(i);
  if (label == 0) {
    TICK_ERROR("Labels 0 should not be considered in SDCA");
  }

  double normalized_features_norm = features_norm_sq[i] / (l_l2sq * n_non_zeros_labels);
  if (use_intercept()) {
    normalized_features_norm += 1. / (l_l2sq * n_non_zeros_labels);
  }
  const double primal_dot_features = get_inner_prod(i, primal_vector);

  const double tmp = dual_i * normalized_features_norm - primal_dot_features;
  double new_dual = (std::sqrt(tmp * tmp + 4 * label * normalized_features_norm) + tmp);
  new_dual /= 2 * normalized_features_norm;

  return new_dual - dual_i;
}

void ModelPoisReg::sdca_primal_dual_relation(const double l_l2sq,
                                             const ArrayDouble &dual_vector,
                                             ArrayDouble &out_primal_vector) {
  if (link_type == LinkType::exponential) {
    ModelGeneralizedLinear::sdca_primal_dual_relation(l_l2sq, dual_vector, out_primal_vector);
    return;
  }

  if (!ready_non_zero_label_map) init_non_zero_label_map();

  if (dual_vector.size() != n_non_zeros_labels) {
    TICK_ERROR("dual vector should have shape of (" << n_non_zeros_labels << ", )");
  }
  if (out_primal_vector.size() != get_n_coeffs()) {
    TICK_ERROR("primal vector should have shape of (" << get_n_coeffs() << ", )");
  }

  const double _1_over_lbda_n = 1 / (l_l2sq * n_non_zeros_labels);
  out_primal_vector.init_to_zero();

  ulong n_non_zero_labels_seen = 0;
  for (ulong i = 0; i < n_samples; ++i) {
    const BaseArrayDouble feature_i = get_features(i);

    double factor;
    if (get_label(i) != 0) {
      const double dual_i = dual_vector[n_non_zero_labels_seen];
      factor = (dual_i - 1) * _1_over_lbda_n;
      n_non_zero_labels_seen += 1;
    } else {
      factor = -_1_over_lbda_n;
    }

    if (fit_intercept) {
      // The last coefficient of out_primal_vector is the intercept
      ArrayDouble w = view(out_primal_vector, 0, get_n_coeffs() - 1);
      w.mult_incr(feature_i, factor);
      out_primal_vector[get_n_coeffs() - 1] += factor;
    } else {
      out_primal_vector.mult_incr(feature_i, factor);
    }
  }
}

double ModelPoisReg::loss_i(const ulong i, const ArrayDouble &coeffs) {
  const double z = get_inner_prod(i, coeffs);
  switch (link_type) {
    case LinkType::exponential: {
      double y_i = get_label(i);
      return exp(z) - y_i * z + std::lgamma(y_i + 1);
    }
    case LinkType::identity: {
      double y_i = get_label(i);
      return z - y_i * log(z) + std::lgamma(y_i + 1);
    }
    default:throw std::runtime_error("Undefined link type");
  }
}

double ModelPoisReg::grad_i_factor(const ulong i, const ArrayDouble &coeffs) {
  const double z = get_inner_prod(i, coeffs);
  switch (link_type) {
    case LinkType::exponential: {
      return exp(z) - get_label(i);
    }
    case LinkType::identity: {
      return 1 - get_label(i) / z;
    }
    default:throw std::runtime_error("Undefined link type");
  }
}
