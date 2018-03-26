// License: BSD 3 clause

#include "tick/prox/prox_l2sq.h"

// Compute the prox on the i-th coordinate only
template <class T>
T TProxL2Sq<T>::call_single(T x, T step) const {
  if (positive && x < 0) {
    return 0;
  } else {
    return x / (1 + step * strength);
  }
}

// Repeat n_times the prox on coordinate i
template <class T>
T TProxL2Sq<T>::call_single(T x, T step, ulong n_times) const {
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

template <class T>
T TProxL2Sq<T>::value_single(T x) const {
  return x * x / 2;
}

template class DLL_PUBLIC TProxL2Sq<double>;
template class DLL_PUBLIC TProxL2Sq<float>;
