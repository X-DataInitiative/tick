#ifndef LIB_INCLUDE_TICK_PROX_PROX_L1_H_
#define LIB_INCLUDE_TICK_PROX_PROX_L1_H_

// License: BSD 3 clause

#include "prox_separable.h"

class ProxL1 : public ProxSeparable {
 public:
  ProxL1(double strength, bool positive);

  ProxL1(double strength, ulong start, ulong end, bool positive);

  const std::string get_class_name() const override;

 private:
  double call_single(double x, double step) const override;

  // Repeat n_times the prox on coordinate i
  double call_single(double x, double step, ulong n_times) const override;

  double value_single(double x) const override;
};

#endif  // LIB_INCLUDE_TICK_PROX_PROX_L1_H_
