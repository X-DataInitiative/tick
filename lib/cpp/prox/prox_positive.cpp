// License: BSD 3 clause

#include "tick/prox/prox_positive.h"

template <class T>
TProxPositive<T>::TProxPositive(T strength)
    : TProxSeparable<T>(strength, true) {}

template <class T>
TProxPositive<T>::TProxPositive(T strength, ulong start, ulong end)
    : TProxSeparable<T>(strength, start, end, true) {}

template <class T>
T TProxPositive<T>::call_single(T x, T step) const {
  if (x < 0) {
    return 0;
  } else {
    return x;
  }
}

template <class T>
T TProxPositive<T>::call_single(T x, T step, ulong n_times) const {
  return call_single(x, step);
}

template <class T>
T TProxPositive<T>::value(const Array<T> &coeffs, ulong start, ulong end) {
  return 0.;
}

template class DLL_PUBLIC TProxPositive<double>;
template class DLL_PUBLIC TProxPositive<float>;
