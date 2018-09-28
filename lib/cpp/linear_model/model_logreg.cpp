// License: BSD 3 clause

//
// Created by Martin Bompaire on 21/10/15.
//

#include "tick/linear_model/model_logreg.h"

template <class T, class K>
void TModelLogReg<T, K>::sigmoid(const Array<T> &x, Array<T> &out) {
  for (ulong i = 0; i < x.size(); ++i) {
    out[i] = sigmoid(x[i]);
  }
}

template <class T, class K>
void TModelLogReg<T, K>::logistic(const Array<T> &x, Array<T> &out) {
  for (ulong i = 0; i < x.size(); ++i) {
    out[i] = logistic(x[i]);
  }
}

template <class T, class K>
T TModelLogReg<T, K>::loss_i(const ulong i, const Array<K> &coeffs) {
  double z_i = get_inner_prod(i, coeffs);
  z_i *= get_label(i);
  return logistic(z_i);
}

template <class T, class K>
T TModelLogReg<T, K>::grad_i_factor(const ulong i, const Array<K> &coeffs) {
  // The label in { -1, 1 }
  const T y_i = get_label(i);
  // Contains x_i^T w + b
  const T z_i = get_inner_prod(i, coeffs);

  return y_i * (sigmoid(y_i * z_i) - 1);
}

template <class T, class K>
T TModelLogReg<T, K>::sdca_dual_min_i(const ulong i, const T dual_i,
                                      const T primal_dot_features,
                                      const T previous_delta_dual_i, T l_l2sq) {
  compute_features_norm_sq();
  T epsilon = 1e-1;
  T normalized_features_norm = features_norm_sq[i] / (l_l2sq * n_samples);
  if (use_intercept()) {
    normalized_features_norm += 1. / (l_l2sq * n_samples);
  }
  const T label = get_label(i);
  T new_dual_times_label{0.};

  // initial delta dual as suggested in original paper
  // http://www.jmlr.org/papers/volume14/shalev-shwartz13a/shalev-shwartz13a.pdf 6.2
  T delta_dual = label / (1. + exp(primal_dot_features * label)) - dual_i;
  delta_dual /= std::max(1., 0.25 + normalized_features_norm);

  for (int j = 0; j < 10; ++j) {
    T new_dual = dual_i + delta_dual;
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
    T f_prime =
        label * (log(new_dual_times_label) - log(1 - new_dual_times_label));
    T f_second = 1 / (new_dual_times_label * (1 - new_dual_times_label));

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


template <class T, class K>
Array<T> TModelLogReg<T, K>::sdca_dual_min_many(ulong n_indices,
                                                const Array<T> &duals,
                                                double l_l2sq,
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

  for (int k = 0; k < 20; ++k) {

    // Check we are in the correct bounds
    for (ulong i = 0; i < n_indices; ++i) {
      new_duals[i] = duals[i] + delta_duals[i];
      T new_dual_times_label_i = sdca_labels[i] * new_duals[i];

      if (new_dual_times_label_i <= 0) {
        T label = sdca_labels[i];
        new_duals[i] = epsilon / label;
        delta_duals[i] = new_duals[i] - duals[i];
        epsilon *= 0.3;
      }
      if (new_dual_times_label_i >= 1) {
        T label = sdca_labels[i];
        new_duals[i] = (1 - epsilon) / label;
        delta_duals[i] = new_duals[i] - duals[i];
        epsilon *= 0.3;
      }
    }

    // Use the opposite to have a positive definite matrix
    for (ulong i = 0; i < n_indices; ++i) {
      T new_dual_times_label_i = sdca_labels[i] * new_duals[i];

      n_grad[i] = sdca_labels[i] * (log(new_dual_times_label_i) - log(1 - new_dual_times_label_i))
          + p[i];

      for (ulong j = 0; j < n_indices; ++j) {
        n_grad[i] += delta_duals[j] * g(i, j);
        n_hess(i, j) = g(i, j);
      }

      n_hess(i, i) += 1 / (new_dual_times_label_i * (1 - new_dual_times_label_i));
    }

    // it seems faster this way with BLAS
    if (n_indices <= 30) {
      tick::vector_operations<T>{}.solve_linear_system(
          n_indices, n_hess.data(), n_grad.data(), ipiv.data());
    } else {
      tick::vector_operations<T>{}.solve_symmetric_linear_system(
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
    new_duals[i] = duals[i] + delta_duals[i];
    T new_dual_times_label_i = sdca_labels[i] * new_duals[i];

    if (new_dual_times_label_i <= 0) {
      T label = sdca_labels[i];
      new_duals[i] = epsilon / label;
      delta_duals[i] = new_duals[i] - duals[i];
    }
    if (new_dual_times_label_i >= 1) {
      T label = sdca_labels[i];
      new_duals[i] = (1 - epsilon) / label;
      delta_duals[i] = new_duals[i] - duals[i];
    }
  }

  return delta_duals;
};


template <class T, class K>
void TModelLogReg<T, K>::compute_lip_consts() {
  if (ready_lip_consts) {
    return;
  } else {
    compute_features_norm_sq();
    lip_consts = Array<T>(n_samples);
    for (ulong i = 0; i < n_samples; ++i) {
      if (fit_intercept) {
        lip_consts[i] = (features_norm_sq[i] + 1) / 4;
      } else {
        lip_consts[i] = features_norm_sq[i] / 4;
      }
    }
  }
}

template class DLL_PUBLIC TModelLogReg<double, double>;
template class DLL_PUBLIC TModelLogReg<float, float>;

template class DLL_PUBLIC TModelLogReg<double, std::atomic<double>>;
template class DLL_PUBLIC TModelLogReg<float, std::atomic<float>>;
