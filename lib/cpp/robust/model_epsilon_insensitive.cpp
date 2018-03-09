// License: BSD 3 clause

#include "tick/robust/model_epsilon_insensitive.h"

template <class T>
TModelEpsilonInsensitive<T>::TModelEpsilonInsensitive(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels, const bool fit_intercept,
    const T threshold, const int n_threads)
    : TModelLabelsFeatures<T>(features, labels),
      TModelGeneralizedLinear<T>(features, labels, fit_intercept, n_threads) {
  set_threshold(threshold);
}

template <class T>
T TModelEpsilonInsensitive<T>::loss_i(const ulong i, const Array<T> &coeffs) {
  const T z = std::abs(get_inner_prod(i, coeffs) - get_label(i));
  if (z > threshold) {
    return z - threshold;
  } else {
    return 0.;
  }
}

template <class T>
T TModelEpsilonInsensitive<T>::grad_i_factor(const ulong i,
                                             const Array<T> &coeffs) {
  const T d = get_inner_prod(i, coeffs) - get_label(i);
  if (std::abs(d) > threshold) {
    if (d > 0) {
      return 1;
    } else {
      return -1;
    }
  } else {
    return 0.;
  }
}

template class DLL_PUBLIC TModelEpsilonInsensitive<double>;
template class DLL_PUBLIC TModelEpsilonInsensitive<float>;
