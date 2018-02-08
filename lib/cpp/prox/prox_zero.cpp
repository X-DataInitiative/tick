// License: BSD 3 clause

#include "tick/prox/prox_zero.h"

template <class T>
TProxZero<T>::TProxZero(T strength) : TProxSeparable<T>(strength, false) {}

template <class T>
TProxZero<T>::TProxZero(T strength, ulong start, ulong end)
    : TProxSeparable<T>(strength, start, end, false) {}

template <class T>
std::string TProxZero<T>::get_class_name() const {
  return "TProxZero<T>";
}

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

template class TProxZero<double>;
template class TProxZero<float>;
