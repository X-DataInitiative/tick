// License: BSD 3 clause

#include "tick/prox/prox_l1.h"

template <class T>
T TProxL1<T>::call_single(T x, T step) const {
  T thresh = step * strength;
  if (x > 0) {
    if (x > thresh) {
      return x - thresh;
    } else {
      return 0;
    }
  } else {
    // If x is negative and we project onto the non-negative half-plane
    // we set it to 0
    if (positive) {
      return 0;
    } else {
      if (x < -thresh) {
        return x + thresh;
      } else {
        return 0;
      }
    }
  }
}

template <class T>
T TProxL1<T>::call_single(T x, T step, ulong n_times) const {
  if (n_times >= 1) {
    return call_single(x, n_times * step);
  } else {
    return x;
  }
}

template <class T>
T TProxL1<T>::value_single(T x) const {
  return std::abs(x);
}

template class DLL_PUBLIC TProxL1<double>;
template class DLL_PUBLIC TProxL1<float>;
