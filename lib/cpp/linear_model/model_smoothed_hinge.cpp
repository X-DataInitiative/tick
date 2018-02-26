// License: BSD 3 clause

#include "tick/linear_model/model_smoothed_hinge.h"

template <class T>
TModelSmoothedHinge<T>::TModelSmoothedHinge(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels, const bool fit_intercept,
    const T smoothness, const int n_threads)
    : TModelLabelsFeatures<T>(features, labels),
      TModelGeneralizedLinear<T>(features, labels, fit_intercept, n_threads) {
  set_smoothness(smoothness);
}

template <class T>
T TModelSmoothedHinge<T>::loss_i(const ulong i, const Array<T> &coeffs) {
  const double z = get_label(i) * get_inner_prod(i, coeffs);
  if (z >= 1) {
    return 0.;
  } else {
    if (z <= 1 - smoothness) {
      return 1 - z - smoothness / 2;
    } else {
      const double d = (1 - z);
      return d * d / (2 * smoothness);
    }
  }
}

template <class T>
T TModelSmoothedHinge<T>::grad_i_factor(const ulong i, const Array<T> &coeffs) {
  const double y = get_label(i);
  const double z = y * get_inner_prod(i, coeffs);
  if (z >= 1) {
    return 0.;
  } else {
    if (z <= 1 - smoothness) {
      return -y;
    } else {
      return (z - 1) * y / smoothness;
    }
  }
}

template <class T>
void TModelSmoothedHinge<T>::compute_lip_consts() {
  if (ready_lip_consts) {
    return;
  } else {
    compute_features_norm_sq();
    lip_consts = Array<T>(n_samples);
    for (ulong i = 0; i < n_samples; ++i) {
      if (fit_intercept) {
        lip_consts[i] = (features_norm_sq[i] + 1) / smoothness;
      } else {
        lip_consts[i] = features_norm_sq[i] / smoothness;
      }
    }
  }
}

template class TModelSmoothedHinge<double>;
template class TModelSmoothedHinge<float>;
