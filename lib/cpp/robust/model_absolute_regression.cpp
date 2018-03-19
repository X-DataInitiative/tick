// License: BSD 3 clause

#include "tick/robust/model_absolute_regression.h"

template <class T>
T TModelAbsoluteRegression<T>::loss_i(const ulong i, const Array<T> &coeffs) {
  return std::abs(get_inner_prod(i, coeffs) - get_label(i));
}

template <class T>
T TModelAbsoluteRegression<T>::grad_i_factor(const ulong i,
                                             const Array<T> &coeffs) {
  const T d = get_inner_prod(i, coeffs) - get_label(i);
  if (d > 0) {
    return 1;
  } else {
    if (d < 0) {
      return -1;
    } else {
      return 0;
    }
  }
}

template class DLL_PUBLIC TModelAbsoluteRegression<double>;
template class DLL_PUBLIC TModelAbsoluteRegression<float>;
