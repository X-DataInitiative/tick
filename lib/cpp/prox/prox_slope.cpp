// License: BSD 3 clause

#include "tick/prox/prox_slope.h"

template <class T>
TProxSlope<T>::TProxSlope(T strength, T false_discovery_rate, bool positive)
    : TProxSortedL1<T>(strength, WeightsType::bh, positive) {
  this->false_discovery_rate = false_discovery_rate;
}

template <class T>
TProxSlope<T>::TProxSlope(T strength, T false_discovery_rate, ulong start,
                          ulong end, bool positive)
    : TProxSortedL1<T>(strength, WeightsType::bh, start, end, positive) {
  this->false_discovery_rate = false_discovery_rate;
}

template <class T>
std::string TProxSlope<T>::get_class_name() const {
  return "TProxSlope";
}

template <class T>
void TProxSlope<T>::compute_weights(void) {
  if (!weights_ready) {
    ulong size = end - start;
    weights = Array<T>(size);
    for (ulong i = 0; i < size; i++) {
      T tmp = false_discovery_rate / (2 * size);
      weights[i] = strength * standard_normal_inv_cdf(1 - tmp * (i + 1));
    }
    weights_ready = true;
  }
}

template class DLL_PUBLIC TProxSlope<double>;
template class DLL_PUBLIC TProxSlope<float>;
