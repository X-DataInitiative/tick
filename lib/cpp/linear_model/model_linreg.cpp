// License: BSD 3 clause

//
// Created by Stéphane GAIFFAS on 12/12/2015.
//

#include "tick/linear_model/model_linreg.h"

template <class T>
T TModelLinReg<T>::sdca_dual_min_i(const ulong i, const T dual_i,
                                   const Array<T> &primal_vector,
                                   const T previous_delta_dual_i, T l_l2sq) {
  compute_features_norm_sq();
  T normalized_features_norm = features_norm_sq[i] / (l_l2sq * n_samples);
  if (use_intercept()) {
    normalized_features_norm += 1. / (l_l2sq * n_samples);
  }
  const T primal_dot_features = get_inner_prod(i, primal_vector);
  const T label = get_label(i);
  const T delta_dual =
      -(dual_i + primal_dot_features - label) / (1 + normalized_features_norm);
  return delta_dual;
}

template <class T>
T TModelLinReg<T>::loss_i(const ulong i, const Array<T> &coeffs) {
  // Compute x_i^T \beta + b
  const T z = get_inner_prod(i, coeffs);
  const T d = get_label(i) - z;
  return d * d / 2;
}

template <class T>
T TModelLinReg<T>::grad_i_factor(const ulong i, const Array<T> &coeffs) {
  const T z = get_inner_prod(i, coeffs);
  return z - get_label(i);
}

template <class T>
void TModelLinReg<T>::compute_lip_consts() {
  if (ready_lip_consts) {
    return;
  } else {
    compute_features_norm_sq();
    lip_consts = Array<T>(n_samples);
    for (ulong i = 0; i < n_samples; ++i) {
      if (fit_intercept) {
        lip_consts[i] = features_norm_sq[i] + 1;
      } else {
        lip_consts[i] = features_norm_sq[i];
      }
    }
  }
}

template class DLL_PUBLIC TModelLinReg<double>;
template class DLL_PUBLIC TModelLinReg<float>;
