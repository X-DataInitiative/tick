// License: BSD 3 clause

#include "tick/linear_model/model_hinge.h"

template <class T>
T TModelHinge<T>::loss_i(const ulong i, const Array<T> &coeffs) {
  const T z = get_label(i) * get_inner_prod(i, coeffs);
  if (z <= 1.) {
    return 1 - z;
  } else {
    return 0.;
  }
}

template <class T>
T TModelHinge<T>::grad_i_factor(const ulong i, const Array<T> &coeffs) {
  const T y = get_label(i);
  const T z = y * get_inner_prod(i, coeffs);
  if (z <= 1.) {
    return -y;
  } else {
    return 0;
  }
}

template class DLL_PUBLIC TModelHinge<double>;
template class DLL_PUBLIC TModelHinge<float>;
