// License: BSD 3 clause

#include "tick/prox/prox_elasticnet.h"

template <class T, class K>
T TProxElasticNet<T, K>::call_single(T x, T step) const {
  T thresh = step * ratio * strength;
  if (x > 0) {
    if (x > thresh) {
      return (x - thresh) / (1 + step * strength * (1 - ratio));
    } else {
      return 0;
    }
  } else {
    // If x is negative and we project onto the non-negative half-plane
    // we set it to 0
    if (positive) {
      return 0;
    } else {
      if (x < -thresh) {
        return (x + thresh) / (1 + step * strength * (1 - ratio));
      } else {
        return 0;
      }
    }
  }
  return 0;
}

template <class T, class K>
T TProxElasticNet<T, K>::value_single(T x) const {
  return (1 - ratio) * 0.5 * x * x + ratio * std::abs(x);
}

template <class T, class K>
T TProxElasticNet<T, K>::get_ratio() const {
  return ratio;
}

template <class T, class K>
void TProxElasticNet<T, K>::set_ratio(T ratio) {
  if (ratio < 0 || ratio > 1)
    TICK_ERROR("Ratio should be in the [0, 1] interval");
  this->ratio = ratio;
}

template class DLL_PUBLIC TProxElasticNet<double, double>;
template class DLL_PUBLIC TProxElasticNet<float, float>;

template class DLL_PUBLIC TProxElasticNet<double, std::atomic<double>>;
template class DLL_PUBLIC TProxElasticNet<float, std::atomic<float>>;
