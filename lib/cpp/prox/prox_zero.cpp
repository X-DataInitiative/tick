// License: BSD 3 clause

#include "tick/prox/prox_zero.h"

template <class T, class K>
T TProxZero<T, K>::call_single(T x, T step) const {
  return x;
}

template <class T, class K>
T TProxZero<T, K>::call_single(T x, T step, ulong n_times) const {
  return x;
}

template <class T, class K>
T TProxZero<T, K>::value(const Array<K> &coeffs, ulong start, ulong end) {
  return 0.;
}

template class DLL_PUBLIC TProxZero<double, double>;
template class DLL_PUBLIC TProxZero<float, float>;

template class DLL_PUBLIC TProxZero<double, std::atomic<double>>;
template class DLL_PUBLIC TProxZero<float, std::atomic<float>>;
