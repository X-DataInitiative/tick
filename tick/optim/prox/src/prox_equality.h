#ifndef TICK_OPTIM_PROX_SRC_PROX_EQUALITY_H_
#define TICK_OPTIM_PROX_SRC_PROX_EQUALITY_H_

// License: BSD 3 clause

#include "prox.h"

class ProxEquality : public Prox {
 public:
  explicit ProxEquality(double strength, bool positive);

  ProxEquality(double strength, ulong start, ulong end, bool positive);

  const std::string get_class_name() const override;

  double value(const ArrayDouble &coeffs, ulong start, ulong end) override;

  void call(const ArrayDouble &coeffs, double step, ArrayDouble &out,
            ulong start, ulong end) override;
};

#endif  // TICK_OPTIM_PROX_SRC_PROX_EQUALITY_H_
