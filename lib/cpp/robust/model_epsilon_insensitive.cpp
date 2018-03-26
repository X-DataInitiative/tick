// License: BSD 3 clause

#include "tick/robust/model_epsilon_insensitive.h"

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
