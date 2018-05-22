// License: BSD 3 clause

#include "tick/robust/model_modified_huber.h"

template <class T, class K>
T TModelModifiedHuber<T, K>::loss_i(const ulong i, const Array<K> &coeffs) {
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

template <class T, class K>
T TModelModifiedHuber<T, K>::grad_i_factor(const ulong i,
                                           const Array<K> &coeffs) {
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

template <class T, class K>
void TModelModifiedHuber<T, K>::compute_lip_consts() {
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

// template class DLL_PUBLIC TModelModifiedHuber<double, std::atomic<double>>;
// template class DLL_PUBLIC TModelModifiedHuber<float, std::atomic<float>>;
