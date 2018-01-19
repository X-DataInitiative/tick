#ifndef LIB_INCLUDE_TICK_PROX_PROX_SLOPE_H_
#define LIB_INCLUDE_TICK_PROX_PROX_SLOPE_H_

// License: BSD 3 clause

#include "prox.h"
#include "prox_sorted_l1.h"

class ProxSlope : public ProxSortedL1 {
 protected:
  double false_discovery_rate;
  void compute_weights(void) override;

 public:
  ProxSlope(double strength, double false_discovery_rate, bool positive);

  ProxSlope(double strength,
            double false_discovery_rate,
            ulong start,
            ulong end,
            bool positive);

  const std::string get_class_name() const override;

  inline double get_false_discovery_rate() const {
    return false_discovery_rate;
  }

  inline void set_false_discovery_rate(double false_discovery_rate) {
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

#endif  // LIB_INCLUDE_TICK_PROX_PROX_SLOPE_H_
