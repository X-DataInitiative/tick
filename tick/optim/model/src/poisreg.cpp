// License: BSD 3 clause

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

std::tuple<double, double> ModelPoisReg::sdca_dual_min_ij(
  const ulong i, const ulong j, const double dual_i, const double dual_j,
  const ArrayDouble &primal_vector, double l_l2sq) {
  if (link_type == LinkType::exponential) {
    TICK_ERROR("Not available for exponential link")
  }

  if (!ready_features_norm_sq) {
    compute_features_norm_sq();
  }

  const double label_i = get_label(i);
  const double label_j = get_label(j);
  if (label_i * label_j == 0) {
    TICK_ERROR("Labels 0 should not be considered in SDCA");
  }

  const double _1_lambda_n = 1 / (l_l2sq * n_non_zeros_labels);
  double g_ii = features_norm_sq[i] * _1_lambda_n;
  double g_jj = features_norm_sq[j] * _1_lambda_n;
  double g_ij = get_features(i).dot(get_features(j)) * _1_lambda_n;

  // if vector are colinear
  if (g_ij * g_ij == g_ii * g_jj) {
    return {0., 0.};
  }

  if (use_intercept()) {
    g_ii += _1_lambda_n;
    g_jj += _1_lambda_n;
    g_ij += _1_lambda_n;
  }

  const double p_i = get_inner_prod(i, primal_vector);
  const double p_j = get_inner_prod(j, primal_vector);

  double delta_dual_i = dual_i == 0 ? 0.1 : 0;
  double delta_dual_j = dual_j == 0 ? 0.1 : 0;
  double epsilon = 1e-1;

  double new_dual_i, new_dual_j;
  double newton_descent_i, newton_descent_j;
  for (int k = 0; k < 20; ++k) {
    new_dual_i = dual_i + delta_dual_i;
    new_dual_j = dual_j + delta_dual_j;

    // Check we are in the correct bounds
    if (new_dual_i <= 0) {
      new_dual_i = epsilon;
      delta_dual_i = new_dual_i - dual_i;
      epsilon *= 1e-1;
    }
    if (new_dual_j <= 0) {
      new_dual_j = epsilon;
      delta_dual_j = new_dual_j - dual_j;
      epsilon *= 1e-1;
    }

    const double n_grad_i = label_i / new_dual_i - p_i - delta_dual_i * g_ii - delta_dual_j * g_ij;
    const double n_grad_j = label_j / new_dual_j - p_j - delta_dual_j * g_jj - delta_dual_i * g_ij;

    const double n_hess_ii = -label_i / (new_dual_i * new_dual_i) - g_ii;
    const double n_hess_jj = -label_j / (new_dual_j * new_dual_j) - g_jj;
    const double n_hess_ij = -g_ij;

//    const double n2_det = n_hess_ii * n_hess_jj - n_hess_ij * n_hess_ij;
//    const double _1_over_n2_det = 1. / n2_det;

//    const double inverse_hess_ii_over_n = _1_over_n2_det * n_hess_jj;
//    const double inverse_hess_jj_over_n = _1_over_n2_det * n_hess_ii;
//    const double inverse_hess_ij_over_n = -_1_over_n2_det * n_hess_ij;

//    newton_descent_i =
//      n_grad_i * inverse_hess_ii_over_n + n_grad_j * inverse_hess_ij_over_n;
//    newton_descent_j =
//      n_grad_i * inverse_hess_ij_over_n + n_grad_j * inverse_hess_jj_over_n;

    double b[2]{n_grad_i, n_grad_j};
    double A[4]{n_hess_ii, n_hess_ij, n_hess_ij, n_hess_jj};

//    tick::vector_operations<double>{}.solve_symmetric_linear_system(2, A, b);
    tick::vector_operations<double>{}.solve_linear_system(2, A, b);

    newton_descent_i = b[0];
    newton_descent_j = b[1];

    delta_dual_i -= newton_descent_i;
    delta_dual_j -= newton_descent_j;

    if (std::abs(newton_descent_i) < 1e-10 && std::abs(newton_descent_j) < 1e-10) {
      break;
    }
  }

  if (std::abs(newton_descent_i) > 1e-4 || std::abs(newton_descent_j) > 1e-4) {

    std::cout << "did not converge newton_descent_i=" << newton_descent_i
              << ", newton_descent_j" << newton_descent_j << "i, j = " << i << " " << j
              << std::endl;
  }

  if (new_dual_i <= 0) {
    new_dual_i = epsilon;
    delta_dual_i = new_dual_i - dual_i;
  }
  if (new_dual_j <= 0) {
    new_dual_j = epsilon;
    delta_dual_j = new_dual_j - dual_j;
  }

  return {delta_dual_i, delta_dual_j};
}

ArrayDouble ModelPoisReg::sdca_dual_min_many(const ArrayULong indices,
                                             const ArrayDouble duals,
                                             const ArrayDouble &primal_vector,
                                             double l_l2sq) {
  const double smallest_dual = 1e-13;

  if (link_type == LinkType::exponential) {
    TICK_ERROR("Not available for exponential link")
  }

  if (!ready_features_norm_sq) {
    compute_features_norm_sq();
  }

  const ulong n_indices = indices.size();

  if (p.size() != n_indices) {
    p = ArrayDouble(n_indices);
    g = ArrayDouble2d(n_indices, n_indices);
    sdca_labels = ArrayDouble(n_indices);

    n_grad = ArrayDouble(n_indices);
    n_hess = ArrayDouble2d(n_indices, n_indices);

    new_duals = ArrayDouble(n_indices);
    delta_duals = ArrayDouble(n_indices);

    ipiv = ArrayInt(n_indices);
  }

  for (ulong i = 0; i < n_indices; ++i) sdca_labels[i] = get_label(indices[i]);
  if (sdca_labels.min() == 0) {
    indices.print();
    sdca_labels.print();
    TICK_ERROR("Labels 0 should not be considered in SDCA");
  }

  const double _1_lambda_n = 1 / (l_l2sq * n_non_zeros_labels);
  for (ulong i = 0; i < n_indices; ++i) {
    for (ulong j = 0; j < n_indices; ++j) {
      if (j < i) g(i, j) = g(j, i);
      else if (i == j) g(i, i) = features_norm_sq[indices[i]] * _1_lambda_n;
      else g(i, j) = get_features(indices[i]).dot(get_features(indices[j])) * _1_lambda_n;
      if (use_intercept()) g(i, j) += _1_lambda_n;
    }
  }

  for (ulong i = 0; i < n_indices; ++i) {
    for (ulong j = i + 1; j < n_indices; ++j) {
      // if vector are colinear
      if (g(i, j) * g(i, j) == g(i, i) * g(j, j)) {
        ArrayDouble delta_duals(n_indices);
        delta_duals.init_to_zero();
        std::cout << "skip colinear" << std::endl;
        indices.print();
        return delta_duals;
      }
    }
  }

  for (ulong i = 0; i < n_indices; ++i) p[i] = get_inner_prod(indices[i], primal_vector);
  for (ulong i = 0; i < n_indices; ++i) delta_duals[i] = duals[i] == 0 ? 0.1 : 0;

  double epsilon = 1e-1;

  for (int k = 0; k < 20; ++k) {

    for (ulong i = 0; i < n_indices; ++i) {
      new_duals[i] = duals[i] + delta_duals[i];

      // Check we are in the correct bounds
      if (new_duals[i] <= smallest_dual) {
        new_duals[i] = epsilon;
        delta_duals[i] = new_duals[i] - duals[i];
        epsilon = std::max(smallest_dual , epsilon * 0.3);
      }
    }

    for (ulong i = 0; i < n_indices; ++i) {
      n_grad[i] = sdca_labels[i] / new_duals[i] - p[i];

      for (ulong j = 0; j < n_indices; ++j) {
        n_grad[i] -= delta_duals[j] * g(i, j);
        n_hess(i, j) = -g(i, j);
      }

      n_hess(i, i) -= sdca_labels[i] / (new_duals[i] * new_duals[i]);
    }

    tick::vector_operations<double>{}.solve_linear_system(n_indices,
                                                          n_hess.data(), n_grad.data(),
                                                          ipiv.data());

    delta_duals.mult_incr(n_grad, -1.);

    bool all_converged = true;
    for (ulong i = 0; i < n_indices; ++i) {
      all_converged &= std::abs(n_grad[i]) < 1e-10;
    }
    if (all_converged) {
      break;
    }
  }

  double mean = 0;
  for (ulong i = 0; i < n_indices; ++i) {
    const double abs_grad_i = std::abs(n_grad[i]);
    mean += abs_grad_i;
  }
  mean /= n_indices;

  if (mean > 1e-4) std::cout << "did not converge with mean=" << mean << std::endl;

  for (ulong i = 0; i < n_indices; ++i) {
    // Check we are in the correct bounds
    if (new_duals[i] <= smallest_dual) {
      new_duals[i] = epsilon;
      delta_duals[i] = new_duals[i] - duals[i];
    }
  }

  return delta_duals;
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
      double loss = z + std::lgamma(y_i + 1);
      if (y_i != 0) loss -= y_i * log(z);
      return loss;
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
      double y_i = get_label(i);
      return y_i != 0 ? 1 - get_label(i) / z : 1;
    }
    default:throw std::runtime_error("Undefined link type");
  }
}

SArrayDouble2dPtr ModelPoisReg::hessian(ArrayDouble &coeffs) {
  if (link_type == LinkType::exponential) {
    TICK_ERROR("Hessian is not implemented for exponential link")
  }

  ArrayDouble2d hess(get_n_coeffs(), get_n_coeffs());
  hess.init_to_zero();

  for (ulong i = 0; i < n_samples; ++i) {
    BaseArrayDouble feature_i = get_features(i);
    if (feature_i.is_sparse()) TICK_ERROR("hessian is not implemented for sparse data");
    double inner_prod = get_inner_prod(i, coeffs);
    double coeff = (*labels)[i] / (inner_prod * inner_prod);
    for (ulong row = 0; row < n_features; ++row) {
      for (ulong col = 0; col < n_features; ++col) {
        hess(row, col) += coeff * feature_i.value(row) * feature_i.value(col);
      }
      if (fit_intercept) {
        hess(row, n_features) += coeff * feature_i.value(row);
      }
    }
    if (fit_intercept) {
      for (ulong col = 0; col < n_features; ++col) {
        hess(n_features, col) += coeff * feature_i.value(col);
      }
      hess(n_features, n_features) += coeff;
    }
  }

  hess /= n_samples;
  return hess.as_sarray2d_ptr();
}