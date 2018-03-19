// License: BSD 3 clause

#include "tick/prox/prox_slope.h"

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
