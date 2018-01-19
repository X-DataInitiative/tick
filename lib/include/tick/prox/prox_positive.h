#ifndef LIB_INCLUDE_TICK_PROX_PROX_POSITIVE_H_
#define LIB_INCLUDE_TICK_PROX_PROX_POSITIVE_H_

// License: BSD 3 clause

#include "prox_separable.h"

class ProxPositive : public ProxSeparable {
 public:
  explicit ProxPositive(double strength);

  ProxPositive(double strength, ulong start, ulong end);

  const std::string get_class_name() const override;

  // Override value, only this value method should be called
  double value(const ArrayDouble &coeffs, ulong start, ulong end) override;

 private:
  double call_single(double x, double step) const override;

  // Repeat n_times the prox on coordinate i
  double call_single(double x, double step, ulong n_times) const override;
};

#endif  // LIB_INCLUDE_TICK_PROX_PROX_POSITIVE_H_
