// License: BSD 3 clause

#include "tick/prox/prox_zero.h"

template <class T>
T TProxZero<T>::call_single(T x, T step) const {
  return x;
}

template <class T>
T TProxZero<T>::call_single(T x, T step, ulong n_times) const {
  return x;
}

template <class T>
T TProxZero<T>::value(const Array<T> &coeffs, ulong start, ulong end) {
  return 0.;
}

template class DLL_PUBLIC TProxZero<double>;
template class DLL_PUBLIC TProxZero<float>;
