// License: BSD 3 clause

#include "tick/prox/prox_slope.h"

template <class T, class K>
void TProxSlope<T, K>::compute_weights(void) {
  if (!weights_ready) {
    ulong size = end - start;
    weights = Array<T>(size);
    for (ulong i = 0; i < size; i++) {
      // tmp is double as float prevents adequate precision for
      //  standard_normal_inv_cdf
      double tmp = false_discovery_rate / (2 * size);
      weights[i] = strength * standard_normal_inv_cdf(1 - tmp * (i + 1));
    }
    weights_ready = true;
  }
}

template class DLL_PUBLIC TProxSlope<double, double>;
template class DLL_PUBLIC TProxSlope<float, float>;

template class DLL_PUBLIC TProxSlope<double, std::atomic<double>>;
template class DLL_PUBLIC TProxSlope<float, std::atomic<float>>;
