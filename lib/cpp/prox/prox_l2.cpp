// License: BSD 3 clause

#include "tick/prox/prox_l2.h"

template <class T, class K>
void TProxL2<T, K>::call(const Array<K> &coeffs, T step, Array<K> &out,
                         ulong start, ulong end) {
  auto sub_coeffs = view(coeffs, start, end);
  auto sub_out = view(out, start, end);
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

template <class T, class K>
T TProxL2<T, K>::value(const Array<K> &coeffs, ulong start, ulong end) {
  T norm_sq = view(coeffs, start, end).norm_sq();
  return strength * std::sqrt((end - start) * norm_sq);
}

template class DLL_PUBLIC TProxL2<double, double>;
template class DLL_PUBLIC TProxL2<float, float>;

template class DLL_PUBLIC TProxL2<double, std::atomic<double>>;
template class DLL_PUBLIC TProxL2<float, std::atomic<float>>;
