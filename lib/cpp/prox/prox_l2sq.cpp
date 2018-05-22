// License: BSD 3 clause

#include "tick/prox/prox_l2sq.h"

// Compute the prox on the i-th coordinate only
template <class T, class K>
T TProxL2Sq<T, K>::call_single(T x, T step) const {
  if (positive && x < 0) {
    return 0;
  } else {
    return x / (1 + step * strength);
  }
}

// Repeat n_times the prox on coordinate i
template <class T, class K>
T TProxL2Sq<T, K>::call_single(T x, T step, ulong n_times) const {
  if (n_times >= 1) {
    if (positive && x < 0) {
      return 0;
    } else {
      return x / std::pow(1 + step * strength, n_times);
    }
  } else {
    return x;
  }
}

template <class T, class K>
T TProxL2Sq<T, K>::value_single(T x) const {
  return x * x / 2;
}

template class DLL_PUBLIC TProxL2Sq<double, double>;
template class DLL_PUBLIC TProxL2Sq<float, float>;

template class DLL_PUBLIC TProxL2Sq<double, std::atomic<double>>;
template class DLL_PUBLIC TProxL2Sq<float, std::atomic<float>>;
