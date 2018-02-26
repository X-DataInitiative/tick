// License: BSD 3 clause

#include "tick/prox/prox_multi.h"

// TProxMulti can be instantiated with strength=0 only, since TProxMulti's
// strength is not used
template <class T>
TProxMulti<T>::TProxMulti(ProxTPtrVector proxs)
    : TProx<T>(0, false), proxs(proxs) {}

template <class T>
T TProxMulti<T>::value(const Array<T> &coeffs, ulong start, ulong end) {
  T val = 0;
  for (auto prox : proxs) {
    val += prox->value(coeffs);
  }
  return val;
}

template <class T>
void TProxMulti<T>::call(const Array<T> &coeffs, T step, Array<T> &out,
                         ulong start, ulong end) {
  // We need a copy
  Array<T> original_coeffs = coeffs;
  // Note for later: if all are ProxSeparable, we can avoid the copy...
  for (auto prox : proxs) {
    prox->call(original_coeffs, step, out);
    original_coeffs = out;
  }
}

template class DLL_PUBLIC TProxMulti<double>;
template class DLL_PUBLIC TProxMulti<float>;
