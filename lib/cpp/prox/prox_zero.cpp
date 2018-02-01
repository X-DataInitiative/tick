// License: BSD 3 clause

#include "tick/prox/prox_zero.h"

template <class T, class K>
TProxZero<T, K>::TProxZero(K strength)
  : TProxSeparable<T, K>(strength, false) {}

template <class T, class K>
TProxZero<T, K>::TProxZero(K strength,
                   ulong start,
                   ulong end)
  : TProxSeparable<T, K>(strength, start, end, false) {}

ProxZero::ProxZero(double strength)
  : TProxZero<double, double>(strength) {}

ProxZero::ProxZero(double strength,
                   ulong start,
                   ulong end)
  : TProxZero<double, double>(strength, start, end) {}


template <class T, class K>
std::string TProxZero<T, K>::get_class_name() const {
  return "TProxZero<T, K>";
}

std::string ProxZero::get_class_name() const {
  return "ProxZero";
}

template <class T, class K>
K
TProxZero<T, K>::call_single(
  K x,
  K step
) const {
  return x;
}

template <class T, class K>
K
TProxZero<T, K>::call_single(
  K x,
  K step,
  ulong n_times
) const {
  return x;
}

template <class T, class K>
K
TProxZero<T, K>::value(
  const Array<T> &coeffs,
  ulong start,
  ulong end
) {
  return 0.;
}

template class TProxZero<double, double>;
template class TProxZero<float , float>;
