// License: BSD 3 clause

#include "tick/prox/prox_l2.h"

template <class T>
TProxL2<T>::TProxL2(T strength, bool positive) : TProx<T>(strength, positive) {}

template <class T>
TProxL2<T>::TProxL2(T strength, ulong start, ulong end, bool positive)
    : TProx<T>(strength, start, end, positive) {}

template <class T>
void TProxL2<T>::call(const Array<T> &coeffs, T step, Array<T> &out,
                      ulong start, ulong end) {
  Array<T> sub_coeffs = view(coeffs, start, end);
  Array<T> sub_out = view(out, start, end);
  const T thresh = step * strength * std::sqrt(end - start);
  T norm = std::sqrt(sub_coeffs.norm_sq());

  if (norm <= thresh) {
    sub_out.fill(0.);
  } else {
    T t = 1. - thresh / norm;
    sub_out *= t;
  }
  if (positive) {
    for (ulong i = 0; i < sub_out.size(); ++i) {
      if (sub_out[i] < 0) {
        sub_out[i] = 0;
      }
    }
  }
}

template <class T>
T TProxL2<T>::value(const Array<T> &coeffs, ulong start, ulong end) {
  T norm_sq = view(coeffs, start, end).norm_sq();
  return strength * std::sqrt((end - start) * norm_sq);
}

template class DLL_PUBLIC TProxL2<double>;
template class DLL_PUBLIC TProxL2<float>;
