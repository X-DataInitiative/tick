// License: BSD 3 clause

#include "tick/linear_model/model_quadratic_hinge.h"

template <class T>
TModelQuadraticHinge<T>::TModelQuadraticHinge(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels, const bool fit_intercept,
    const int n_threads)
    : TModelLabelsFeatures<T>(features, labels),
      TModelGeneralizedLinear<T>(features, labels, fit_intercept, n_threads) {}

template <class T>
T TModelQuadraticHinge<T>::loss_i(const ulong i, const Array<T> &coeffs) {
  const T z = get_label(i) * get_inner_prod(i, coeffs);
  if (z < 1.) {
    const T d = 1. - z;
    return d * d / 2;
  } else {
    return 0.;
  }
}

template <class T>
T TModelQuadraticHinge<T>::grad_i_factor(const ulong i,
                                         const Array<T> &coeffs) {
  const T y = get_label(i);
  const T z = y * get_inner_prod(i, coeffs);
  if (z < 1) {
    return y * (z - 1);
  } else {
    return 0;
  }
}

template <class T>
void TModelQuadraticHinge<T>::compute_lip_consts() {
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

template class TModelQuadraticHinge<double>;
template class TModelQuadraticHinge<float>;
