#ifndef LIB_INCLUDE_TICK_PROX_PROX_SLOPE_H_
#define LIB_INCLUDE_TICK_PROX_PROX_SLOPE_H_

// License: BSD 3 clause

#include "prox.h"
#include "prox_sorted_l1.h"

template <class T>
class DLL_PUBLIC TProxSlope : public TProxSortedL1<T> {
 protected:
  using TProxSortedL1<T>::start;
  using TProxSortedL1<T>::end;
  using TProxSortedL1<T>::weights_ready;
  using TProxSortedL1<T>::weights;
  using TProxSortedL1<T>::strength;

 protected:
  T false_discovery_rate;
  void compute_weights(void) override;

 public:
  TProxSlope(T strength, T false_discovery_rate, bool positive);

  TProxSlope(T strength, T false_discovery_rate, ulong start, ulong end,
             bool positive);

  std::string get_class_name() const override;

  inline T get_false_discovery_rate() const { return false_discovery_rate; }

  inline void set_false_discovery_rate(T false_discovery_rate) {
    if (false_discovery_rate <= 0 || false_discovery_rate >= 1) {
      TICK_ERROR("False discovery rate must be in (0, 1) but received "
                 << false_discovery_rate)
    }
    if (false_discovery_rate != this->false_discovery_rate) {
      weights_ready = false;
    }
    this->false_discovery_rate = false_discovery_rate;
  }
};

using ProxSlope = TProxSlope<double>;

using ProxSlopeDouble = TProxSlope<double>;
using ProxSlopeFloat = TProxSlope<float>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_SLOPE_H_
