// License: BSD 3 clause

#include "tick/robust/model_absolute_regression.h"

template <class T, class K>
T TModelAbsoluteRegression<T, K>::loss_i(const ulong i,
                                         const Array<K> &coeffs) {
  return std::abs(get_inner_prod(i, coeffs) - get_label(i));
}

template <class T, class K>
T TModelAbsoluteRegression<T, K>::grad_i_factor(const ulong i,
                                                const Array<K> &coeffs) {
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

// template class DLL_PUBLIC TModelAbsoluteRegression<double,
// std::atomic<double>>; template class DLL_PUBLIC
// TModelAbsoluteRegression<float, std::atomic<float>>;
