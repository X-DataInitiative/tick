// License: BSD 3 clause

#include "tick/robust/model_modified_huber.h"

template <class T>
TModelModifiedHuber<T>::TModelModifiedHuber(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels, const bool fit_intercept,
    const int n_threads)
    : TModelLabelsFeatures<T>(features, labels),
      TModelGeneralizedLinear<T>(features, labels, fit_intercept, n_threads) {}

template <class T>
T TModelModifiedHuber<T>::loss_i(const ulong i, const Array<T> &coeffs) {
  const T z = get_label(i) * get_inner_prod(i, coeffs);
  if (z >= 1) {
    return 0.;
  } else {
    if (z <= -1) {
      return -4 * z;
    } else {
      const T d = 1 - z;
      return d * d;
    }
  }
}

template <class T>
T TModelModifiedHuber<T>::grad_i_factor(const ulong i, const Array<T> &coeffs) {
  const T y = get_label(i);
  const T z = y * get_inner_prod(i, coeffs);
  if (z >= 1) {
    return 0.;
  } else {
    if (z <= -1) {
      return -4 * y;
    } else {
      return 2 * y * (z - 1);
    }
  }
}

template <class T>
void TModelModifiedHuber<T>::compute_lip_consts() {
  if (ready_lip_consts) {
    return;
  } else {
    compute_features_norm_sq();
    lip_consts = Array<T>(n_samples);
    for (ulong i = 0; i < n_samples; ++i) {
      if (fit_intercept) {
        lip_consts[i] = 2 * (features_norm_sq[i] + 1);
      } else {
        lip_consts[i] = 2 * features_norm_sq[i];
      }
    }
  }
}

template class DLL_PUBLIC TModelModifiedHuber<double>;
template class DLL_PUBLIC TModelModifiedHuber<float>;
