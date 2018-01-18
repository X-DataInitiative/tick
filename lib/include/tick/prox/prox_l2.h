#ifndef LIB_INCLUDE_TICK_PROX_PROX_L2_H_
#define LIB_INCLUDE_TICK_PROX_PROX_L2_H_

// License: BSD 3 clause

#include "prox.h"

class ProxL2 : public Prox {
 public:
  ProxL2(double strength, bool positive);

  ProxL2(double strength, ulong start, ulong end, bool positive);

  const std::string get_class_name() const override;

  double value(const ArrayDouble &coeffs, ulong start, ulong end) override;

  void call(const ArrayDouble &coeffs, double step, ArrayDouble &out,
            ulong start, ulong end) override;
};

#endif  // LIB_INCLUDE_TICK_PROX_PROX_L2_H_
