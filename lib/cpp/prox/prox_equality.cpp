// License: BSD 3 clause

#include "tick/prox/prox_equality.h"
#include "tick/base/base.h"

template <class T, class K>
T TProxEquality<T, K>::value(const Array<K> &coeffs, ulong start, ulong end) {
  auto sub_coeffs = view(coeffs, start, end);
  if (sub_coeffs.min() == sub_coeffs.max()) {
    return 0;
  } else {
    return std::numeric_limits<T>::max();
  }
}

template <class T, class K>
void TProxEquality<T, K>::call(const Array<K> &coeffs, T step, Array<K> &out,
                               ulong start, ulong end) {
  auto sub_coeffs = view(coeffs, start, end);
  auto sub_out = view(out, start, end);
  T mean = sub_coeffs.sum() / sub_coeffs.size();
  if (positive && (mean < 0)) {
    sub_out.fill(0);
  } else {
    sub_out.fill(mean);
  }
}

template class DLL_PUBLIC TProxEquality<double, double>;
template class DLL_PUBLIC TProxEquality<float, float>;

template class DLL_PUBLIC TProxEquality<double, std::atomic<double>>;
template class DLL_PUBLIC TProxEquality<float, std::atomic<float>>;
