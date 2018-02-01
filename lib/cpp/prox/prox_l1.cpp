// License: BSD 3 clause

#include "tick/prox/prox_l1.h"

template <class T, class K>
TProxL1<T, K>::TProxL1(K strength,
               bool positive)
    : TProxSeparable<T, K>(strength, positive) {}

template <class T, class K>
TProxL1<T, K>::TProxL1(K strength,
               ulong start,
               ulong end,
               bool positive)
    : TProxSeparable<T, K>(strength, start, end, positive) {}


ProxL1::ProxL1(double strength,
               bool positive)
    : TProxL1<double, double>(strength, positive) {}

ProxL1::ProxL1(double strength,
               ulong start,
               ulong end,
               bool positive)
    : TProxL1<double, double>(strength, start, end, positive) {}

template <class T, class K>
std::string
TProxL1<T, K>::get_class_name() const {
  return "TProxL1<T, K>";
}

std::string ProxL1::get_class_name() const {
  return "ProxL1";
}

template <class T, class K>
K
TProxL1<T, K>::call_single(K x,
                           K step) const {
  K thresh = step * strength;
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

template <class T, class K>
K
TProxL1<T, K>::call_single(K x,
                           K step,
                           ulong n_times) const {
  if (n_times >= 1) {
    return call_single(x, n_times * step);
  } else {
    return x;
  }
}

template <class T, class K>
K
TProxL1<T, K>::value_single(K x) const {
  return std::abs(x);
}

template class TProxL1<double, double>;
template class TProxL1<float , float>;
