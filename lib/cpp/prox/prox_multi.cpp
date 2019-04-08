// License: BSD 3 clause

#include "tick/prox/prox_multi.h"

// TProxMulti can be instantiated with strength=0 only, since TProxMulti's
// strength is not used
template <class T, class K>
TProxMulti<T, K>::TProxMulti(ProxTPtrVector proxs)
    : TProx<T, K>(0, false), proxs(proxs) {}

template <class T, class K>
T TProxMulti<T, K>::value(const Array<K> &coeffs, ulong start, ulong end) {
  T val = 0;
  for (auto prox : proxs) {
    val += prox->value(coeffs);
  }
  return val;
}

template <class T, class K>
void TProxMulti<T, K>::call(const Array<K> &coeffs, T step, Array<K> &out,
                            ulong start, ulong end) {
  // We need a copy
  Array<K> original_coeffs = coeffs;
  // Note for later: if all are ProxSeparable, we can avoid the copy...
  for (auto prox : proxs) {
    prox->call(original_coeffs, step, out);
    original_coeffs = out;
  }
}

template class DLL_PUBLIC TProxMulti<double, double>;
template class DLL_PUBLIC TProxMulti<float, float>;

template class DLL_PUBLIC TProxMulti<double, std::atomic<double>>;
template class DLL_PUBLIC TProxMulti<float, std::atomic<float>>;
