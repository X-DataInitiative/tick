// License: BSD 3 clause

#include "tick/prox/prox_positive.h"

template <class T, class K>
T TProxPositive<T, K>::call_single(T x, T step) const {
  if (x < 0) {
    return 0;
  } else {
    return x;
  }
}

template <class T, class K>
T TProxPositive<T, K>::call_single(T x, T step, ulong n_times) const {
  return call_single(x, step);
}

template <class T, class K>
T TProxPositive<T, K>::value(const Array<K> &coeffs, ulong start, ulong end) {
  return 0.;
}

template class DLL_PUBLIC TProxPositive<double, double>;
template class DLL_PUBLIC TProxPositive<float, float>;

template class DLL_PUBLIC TProxPositive<double, std::atomic<double>>;
template class DLL_PUBLIC TProxPositive<float, std::atomic<float>>;
