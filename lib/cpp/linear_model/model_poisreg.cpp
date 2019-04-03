// License: BSD 3 clause

//
// Created by Martin Bompaire on 21/10/15.
//

#include "tick/linear_model/model_poisreg.h"

template <class T, class K>
T TModelPoisReg<T, K>::sdca_dual_min_i(const ulong i, const T dual_i,
                                       const T primal_dot_features,
                                       const T previous_delta_dual_i,
                                       T _1_over_lbda_n) {
  if (link_type == LinkType::identity) {
    return sdca_dual_min_i_identity(i, dual_i, primal_dot_features,
                                    previous_delta_dual_i, _1_over_lbda_n);
  } else {
    return sdca_dual_min_i_exponential(i, dual_i, primal_dot_features,
                                       previous_delta_dual_i, _1_over_lbda_n);
  }
}


template <class T, class K>
Array<T> TModelPoisReg<T, K>::sdca_dual_min_many(ulong n_indices,
                                                 const Array<T> &duals,
                                                 Array2d<T> &g,
                                                 Array2d<T> &n_hess,
                                                 Array<T> &p,
                                                 Array<T> &n_grad,
                                                 Array<T> &sdca_labels,
                                                 Array<T> &new_duals,
                                                 Array<T> &delta_duals,
                                                 ArrayInt &ipiv) {
  if (link_type == LinkType::identity) {
    return sdca_dual_min_many_identity(
        n_indices, duals, g, n_hess, p, n_grad, sdca_labels,
        new_duals, delta_duals, ipiv);
  } else {
    return sdca_dual_min_many_exponential(
        n_indices, duals, g, n_hess, p, n_grad, sdca_labels,
        new_duals, delta_duals, ipiv);
  }
}


template <class T, class K>
T TModelPoisReg<T, K>::sdca_dual_min_i_exponential(
    const ulong i, const T dual_i, const T primal_dot_features,
    const T previous_delta_dual_i, T _1_over_lbda_n) {
  compute_features_norm_sq();
  T epsilon = 1e-1;

  T normalized_features_norm = features_norm_sq[i] * _1_over_lbda_n;
  if (use_intercept()) {
    normalized_features_norm += _1_over_lbda_n;
  }
  T delta_dual = previous_delta_dual_i;
  const T label = get_label(i);
  T new_dual{0.};

  for (size_t j = 0; j < 10; ++j) {
    new_dual = dual_i + delta_dual;

    // Check we are in the correct bounds
    if (new_dual >= label) {
      new_dual = label - epsilon;
      delta_dual = new_dual - dual_i;
      epsilon *= 1e-1;
    }

    // Do newton descent
    // Poisson loss part
    T f_prime = -log(label - new_dual);
    T f_second = 1. / (label - new_dual);

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

template <class T, class K>
Array<T> TModelPoisReg<T, K>::sdca_dual_min_many_exponential(ulong n_indices,
                                                      const Array<T> &duals,
                                                      Array2d<T> &g,
                                                      Array2d<T> &n_hess,
                                                      Array<T> &p,
                                                      Array<T> &n_grad,
                                                      Array<T> &sdca_labels,
                                                      Array<T> &new_duals,
                                                      Array<T> &delta_duals,
                                                      ArrayInt &ipiv) {
  compute_features_norm_sq();

  T epsilon = 1e-1;

  delta_duals.init_to_zero();

  for (size_t k = 0; k < 20; ++k) {
    // Check we are in the correct bounds
    for (ulong i = 0; i < n_indices; ++i) {
      new_duals[i] = duals[i] + delta_duals[i];

      if (new_duals[i] >= sdca_labels[i]) {
        new_duals[i] = sdca_labels[i] - epsilon;
        delta_duals[i] = new_duals[i] - duals[i];
        epsilon *= 0.3;
      }
    }

    // Use the opposite to have a positive definite matrix
    for (ulong i = 0; i < n_indices; ++i) {
      n_grad[i] = -log(sdca_labels[i] - new_duals[i]) + p[i];

      for (ulong j = 0; j < n_indices; ++j) {
        n_grad[i] += delta_duals[j] * g(i, j);
        n_hess(i, j) = g(i, j);
      }

      n_hess(i, i) += 1. / (sdca_labels[i] - new_duals[i]);
    }

    // it seems faster this way with BLAS
    if (n_indices <= 30) {
      tick::vector_operations<T>{}.solve_linear_system(
          n_indices, n_hess.data(), n_grad.data(), ipiv.data());
    } else {
      tick::vector_operations<T>{}.solve_positive_symmetric_linear_system(
          n_indices, n_hess.data(), n_grad.data());
    }

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


  // Check we are in the correct bounds
  for (ulong i = 0; i < n_indices; ++i) {
    if (new_duals[i] >= sdca_labels[i]) {
      new_duals[i] = sdca_labels[i] - epsilon;
      delta_duals[i] = new_duals[i] - duals[i];
    }
  }

  return delta_duals;
}

template <class T, class K>
void TModelPoisReg<T, K>::init_non_zero_label_map() {
  non_zero_labels = VArrayULong::new_ptr();
  for (ulong i = 0; i < get_n_samples(); ++i) {
    if (get_label(i) != 0) {
      non_zero_labels->append1(i);
    }
  }
  n_non_zeros_labels = non_zero_labels->size();
  ready_non_zero_label_map = true;
}

template <class T, class K>
T TModelPoisReg<T, K>::sdca_dual_min_i_identity(const ulong i, const T dual_i,
                                                const T primal_dot_features,
                                                const T previous_delta_dual_i,
                                                T _1_over_lbda_n) {
  if (!ready_features_norm_sq) {
    compute_features_norm_sq();
  }

  const T label = get_label(i);
  if (label == 0) {
    TICK_ERROR("Labels 0 should not be considered in SDCA");
  }

  T normalized_features_norm =
      features_norm_sq[i] * _1_over_lbda_n;
  if (use_intercept()) {
    normalized_features_norm += _1_over_lbda_n;
  }

  const T tmp = dual_i * normalized_features_norm - primal_dot_features;
  T new_dual =
      (std::sqrt(tmp * tmp + 4 * label * normalized_features_norm) + tmp);
  new_dual /= 2 * normalized_features_norm;

  return new_dual - dual_i;
}

template <class T, class K>
Array<T> TModelPoisReg<T, K>::sdca_dual_min_many_identity(ulong n_indices,
                                                             const Array<T> &duals,
                                                             Array2d<T> &g,
                                                             Array2d<T> &n_hess,
                                                             Array<T> &p,
                                                             Array<T> &n_grad,
                                                             Array<T> &sdca_labels,
                                                             Array<T> &new_duals,
                                                             Array<T> &delta_duals,
                                                             ArrayInt &ipiv) {
  compute_features_norm_sq();

  T epsilon = 1e-1;

  delta_duals.init_to_zero();

  for (size_t k = 0; k < 20; ++k) {
    // Check we are in the correct bounds
    for (ulong i = 0; i < n_indices; ++i) {
      if (sdca_labels[i] == 0) {
        TICK_ERROR("Labels 0 should not be considered in SDCA");
      }

      new_duals[i] = duals[i] + delta_duals[i];

      if (new_duals[i] <= 0) {
        new_duals[i] = epsilon;
        delta_duals[i] = new_duals[i] - duals[i];
        epsilon *= 0.3;
      }
    }

    // Use the opposite to have a positive definite matrix
    for (ulong i = 0; i < n_indices; ++i) {
      n_grad[i] = - sdca_labels[i] / new_duals[i] + p[i];

      for (ulong j = 0; j < n_indices; ++j) {
        n_grad[i] += delta_duals[j] * g(i, j);
        n_hess(i, j) = g(i, j);
      }

      n_hess(i, i) += sdca_labels[i] / (new_duals[i] * new_duals[i]);
    }

    // it seems faster this way with BLAS
    if (n_indices <= 30) {
      tick::vector_operations<T>{}.solve_linear_system(
          n_indices, n_hess.data(), n_grad.data(), ipiv.data());
    } else {
      tick::vector_operations<T>{}.solve_positive_symmetric_linear_system(
          n_indices, n_hess.data(), n_grad.data());
    }

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
//  mean /= n_indices;

//  if (mean > 1e-4) std::cout << "did not converge with mean=" << mean << std::endl;

  // Check we are in the correct bounds
  for (ulong i = 0; i < n_indices; ++i) {
    new_duals[i] = duals[i] + delta_duals[i];

    if (new_duals[i] <= 0) {
      new_duals[i] = epsilon;
      delta_duals[i] = new_duals[i] - duals[i];
    }
  }

  return delta_duals;
}


template <class T, class K>
void TModelPoisReg<T, K>::sdca_primal_dual_relation(
    const T _1_over_lbda_n, const Array<K> &dual_vector, Array<K> &out_primal_vector) {
  if (link_type == LinkType::exponential) {
    TModelGeneralizedLinear<T, K>::sdca_primal_dual_relation(
        _1_over_lbda_n, dual_vector, out_primal_vector);
    return;
  }

  if (!ready_non_zero_label_map) init_non_zero_label_map();

  if (dual_vector.size() != n_non_zeros_labels) {
    TICK_ERROR("dual vector should have shape of (" << n_non_zeros_labels
                                                    << ", )");
  }
  if (out_primal_vector.size() != get_n_coeffs()) {
    TICK_ERROR("primal vector should have shape of (" << get_n_coeffs()
                                                      << ", )");
  }

  out_primal_vector.init_to_zero();

  ulong n_non_zero_labels_seen = 0;
  for (ulong i = 0; i < n_samples; ++i) {
    const BaseArray<T> feature_i = get_features(i);

    T factor;
    if (get_label(i) != 0) {
      const T dual_i = dual_vector[n_non_zero_labels_seen];
      factor = (dual_i - 1) * _1_over_lbda_n;
      n_non_zero_labels_seen += 1;
    } else {
      factor = - _1_over_lbda_n;
    }

    if (fit_intercept) {
      // The last coefficient of out_primal_vector is the intercept
      Array<K> w = view(out_primal_vector, 0, get_n_coeffs() - 1);
      w.mult_incr(feature_i, factor);
      out_primal_vector[get_n_coeffs() - 1] = out_primal_vector[get_n_coeffs() - 1] + factor;
    } else {
      out_primal_vector.mult_incr(feature_i, factor);
    }
  }
}

template <class T, class K>
T TModelPoisReg<T, K>::loss_i(const ulong i, const Array<K> &coeffs) {
  const T z = get_inner_prod(i, coeffs);
  switch (link_type) {
    case LinkType::exponential: {
      T y_i = get_label(i);
      return exp(z) - y_i * z + std::lgamma(y_i + 1);
    }
    case LinkType::identity: {
      T y_i = get_label(i);
      if (y_i == 0) return z + std::lgamma(y_i + 1);
      return z + std::lgamma(y_i + 1) - y_i * log(z);
    }
    default:
      throw std::runtime_error("Undefined link type");
  }
}

template <class T, class K>
T TModelPoisReg<T, K>::grad_i_factor(const ulong i, const Array<K> &coeffs) {
  const double z = get_inner_prod(i, coeffs);
  switch (link_type) {
    case LinkType::exponential: {
      return exp(z) - get_label(i);
    }
    case LinkType::identity: {
      return 1 - get_label(i) / z;
    }
    default:
      throw std::runtime_error("Undefined link type");
  }
}

template <class T, class K>
std::shared_ptr<SArray2d<T>> TModelPoisReg<T, K>::hessian(Array<K> &coeffs) {
  if (link_type == LinkType::exponential) {
    TICK_ERROR("Hessian is not implemented for exponential link")
  }

  Array2d<T> hess(get_n_coeffs(), get_n_coeffs());
  hess.init_to_zero();

  for (ulong i = 0; i < n_samples; ++i) {
    BaseArray<T> feature_i = get_features(i);
    if (feature_i.is_sparse()) TICK_ERROR("hessian is not implemented for sparse data");
    T inner_prod = get_inner_prod(i, coeffs);
    T coeff = get_label(i) / (inner_prod * inner_prod);
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

template class DLL_PUBLIC TModelPoisReg<double, double>;
template class DLL_PUBLIC TModelPoisReg<float, float>;

template class DLL_PUBLIC TModelPoisReg<double, std::atomic<double>>;
template class DLL_PUBLIC TModelPoisReg<float, std::atomic<float>>;
