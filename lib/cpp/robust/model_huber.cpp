// License: BSD 3 clause

#include "tick/robust/model_huber.h"

template <class T>
T TModelHuber<T>::loss_i(const ulong i, const Array<T> &coeffs) {
  const T d = get_inner_prod(i, coeffs) - get_label(i);
  const T d_abs = std::abs(d);
  if (d_abs < threshold) {
    return d * d / 2;
  } else {
    return threshold * d_abs - threshold_squared_over_two;
  }
}

template <class T>
T TModelHuber<T>::grad_i_factor(const ulong i, const Array<T> &coeffs) {
  const T d = get_inner_prod(i, coeffs) - get_label(i);
  if (std::abs(d) <= threshold) {
    return d;
  } else {
    if (d >= 0) {
      return threshold;
    } else {
      return -threshold;
    }
  }
}

template <class T>
void TModelHuber<T>::compute_lip_consts() {
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

template class DLL_PUBLIC TModelHuber<double>;
template class DLL_PUBLIC TModelHuber<float>;
