// License: BSD 3 clause

//
// Created by Martin Bompaire on 21/10/15.
//

#include "tick/linear_model/model_poisreg.h"

template <class T, class K>
T TModelPoisReg<T, K>::sdca_dual_min_i(const ulong i, const T dual_i,
                                       const Array<K> &primal_vector,
                                       const T previous_delta_dual_i,
                                       T l_l2sq) {
  if (link_type == LinkType::identity) {
    return sdca_dual_min_i_identity(i, dual_i, primal_vector,
                                    previous_delta_dual_i, l_l2sq);
  } else {
    return sdca_dual_min_i_exponential(i, dual_i, primal_vector,
                                       previous_delta_dual_i, l_l2sq);
  }
}

template <class T, class K>
T TModelPoisReg<T, K>::sdca_dual_min_i_exponential(
    const ulong i, const T dual_i, const Array<K> &primal_vector,
    const T previous_delta_dual_i, T l_l2sq) {
  compute_features_norm_sq();
  T epsilon = 1e-1;

  T normalized_features_norm = features_norm_sq[i] / (l_l2sq * n_samples);
  if (use_intercept()) {
    normalized_features_norm += 1. / (l_l2sq * n_samples);
  }
  const T primal_dot_features = get_inner_prod(i, primal_vector);
  T delta_dual = previous_delta_dual_i;
  const T label = get_label(i);
  T new_dual{0.};

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
                                                const Array<K> &primal_vector,
                                                const T previous_delta_dual_i,
                                                T l_l2sq) {
  if (!ready_features_norm_sq) {
    compute_features_norm_sq();
  }

  const T label = get_label(i);
  if (label == 0) {
    TICK_ERROR("Labels 0 should not be considered in SDCA");
  }

  T normalized_features_norm =
      features_norm_sq[i] / (l_l2sq * n_non_zeros_labels);
  if (use_intercept()) {
    normalized_features_norm += 1. / (l_l2sq * n_non_zeros_labels);
  }
  const T primal_dot_features = get_inner_prod(i, primal_vector);

  const T tmp = dual_i * normalized_features_norm - primal_dot_features;
  T new_dual =
      (std::sqrt(tmp * tmp + 4 * label * normalized_features_norm) + tmp);
  new_dual /= 2 * normalized_features_norm;

  return new_dual - dual_i;
}

template <class T, class K>
void TModelPoisReg<T, K>::sdca_primal_dual_relation(
    const T l_l2sq, const Array<T> &dual_vector, Array<T> &out_primal_vector) {
  if (link_type == LinkType::exponential) {
    TModelGeneralizedLinear<T, K>::sdca_primal_dual_relation(
        l_l2sq, dual_vector, out_primal_vector);
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

  const T _1_over_lbda_n = 1 / (l_l2sq * n_non_zeros_labels);
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
      factor = -_1_over_lbda_n;
    }

    if (fit_intercept) {
      // The last coefficient of out_primal_vector is the intercept
      Array<T> w = view(out_primal_vector, 0, get_n_coeffs() - 1);
      w.mult_incr(feature_i, factor);
      out_primal_vector[get_n_coeffs() - 1] += factor;
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
      return z - y_i * log(z) + std::lgamma(y_i + 1);
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

template class DLL_PUBLIC TModelPoisReg<double, double>;
template class DLL_PUBLIC TModelPoisReg<float, float>;

template class DLL_PUBLIC TModelPoisReg<double, std::atomic<double>>;
template class DLL_PUBLIC TModelPoisReg<float, std::atomic<float>>;
