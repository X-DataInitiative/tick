// License: BSD 3 clause

#include "tick/prox/prox_equality.h"
#include "tick/base/base.h"

template <class T>
T TProxEquality<T>::value(const Array<T> &coeffs, ulong start, ulong end) {
  Array<T> sub_coeffs = view(coeffs, start, end);
  if (sub_coeffs.min() == sub_coeffs.max()) {
    return 0;
  } else {
    return std::numeric_limits<T>::max();
  }
}

template <class T>
void TProxEquality<T>::call(const Array<T> &coeffs, T step, Array<T> &out,
                            ulong start, ulong end) {
  Array<T> sub_coeffs = view(coeffs, start, end);
  Array<T> sub_out = view(out, start, end);
  T mean = sub_coeffs.sum() / sub_coeffs.size();
  if (positive && (mean < 0)) {
    sub_out.fill(0);
  } else {
    sub_out.fill(mean);
  }
}

template class DLL_PUBLIC TProxEquality<double>;
template class DLL_PUBLIC TProxEquality<float>;
