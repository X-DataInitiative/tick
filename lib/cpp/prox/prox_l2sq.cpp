// License: BSD 3 clause

#include "tick/prox/prox_l2sq.h"

template <class T, class K>
TProxL2Sq<T, K>::TProxL2Sq(
  K strength,
  bool positive
) : TProxSeparable<T, K>(strength, positive) {}

template <class T, class K>
TProxL2Sq<T, K>::TProxL2Sq(
  K strength,
  ulong start,
  ulong end,
  bool positive
) : TProxSeparable<T, K>(strength, start, end, positive) {}


ProxL2Sq::ProxL2Sq(
  double strength,
  bool positive
) : TProxL2Sq<double, double>(strength, positive) {}

ProxL2Sq::ProxL2Sq(
  double strength,
  ulong start,
  ulong end,
  bool positive
) : TProxL2Sq<double, double>(strength, start, end, positive)
{}

template <class T, class K>
std::string
TProxL2Sq<T, K>::get_class_name() const {
  return "TProxL2Sq<T, K>";
}

std::string
ProxL2Sq::get_class_name() const {
  return "ProxL2Sq";
}

// Compute the prox on the i-th coordinate only
template <class T, class K>
K
TProxL2Sq<T, K>::call_single(
  K x,
  K step
) const {
  if (positive && x < 0) {
    return 0;
  } else {
    return x / (1 + step * strength);
  }
}

// Repeat n_times the prox on coordinate i
template <class T, class K>
K
TProxL2Sq<T, K>::call_single(
  K x,
  K step,
  ulong n_times
) const {
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
K
TProxL2Sq<T, K>::value_single(K x) const {
  return x * x / 2;
}

template class TProxL2Sq<double, double>;
template class TProxL2Sq<float , float>;
