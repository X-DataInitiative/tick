//
// Created by Martin Bompaire on 21/10/15.
//

#include "poisreg.h"

ModelPoisReg::ModelPoisReg(const SBaseArrayDouble2dPtr features,
                           const SArrayDoublePtr labels,
                           const LinkType link_type,
                           const bool fit_intercept,
                           const int n_threads)
    : ModelGeneralizedLinear(features,
                             labels,
                             fit_intercept,
                             n_threads),
      link_type(link_type) {}

// TODO: Add all the methods for first order computation


double ModelPoisReg::sdca_dual_min_i(const ulong i,
                                     const ArrayDouble &dual_vector,
                                     const ArrayDouble &primal_vector,
                                     const ArrayDouble &previous_delta_dual,
                                     const double l_l2sq) {
  if (link_type == LinkType::identity) {
    throw std::invalid_argument("SDCA not implemented for identity link");
  }

  compute_features_norm_sq();
  double epsilon = 1e-1;

  double normalized_features_norm = features_norm_sq[i] / (l_l2sq * n_samples);
  if (use_intercept()) {
    normalized_features_norm += 1. / (l_l2sq * n_samples);
  }
  const double primal_dot_features = get_inner_prod(i, primal_vector);
  double delta_dual = previous_delta_dual[i];
  const double dual = dual_vector[i];
  const double label = get_label(i);
  double new_dual{0.};

  for (int j = 0; j < 10; ++j) {
    new_dual = dual + delta_dual;

    // Check we are in the correct bounds
    if (new_dual >= label) {
      new_dual = label - epsilon;
      delta_dual = new_dual - dual;
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
    new_dual = dual + delta_dual;

    if (std::abs(f_prime / f_second) < 1e-10) {
      break;
    }
  }
  // Check we are in the correct bounds
  if (new_dual >= label) {
    new_dual = label - epsilon;
    delta_dual = new_dual - dual;
  }

  return delta_dual;
}

double ModelPoisReg::loss_i(const ulong i, const ArrayDouble &coeffs) {
  const double z = get_inner_prod(i, coeffs);
  switch (link_type) {
    case LinkType::exponential: {
      return exp(z) - get_label(i) * z;
    }
    case LinkType::identity: {
      return z - get_label(i) * log(z);
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
