
#ifndef TICK_OPTIM_PROX_SRC_PROX_EQUALITY_H_
#define TICK_OPTIM_PROX_SRC_PROX_EQUALITY_H_

#include "prox.h"

class ProxEquality : public Prox {
 protected:
  bool positive;

 public:
  explicit ProxEquality(double strength, bool positive);

  ProxEquality(double strength, ulong start, ulong end, bool positive);

  const std::string get_class_name() const;

  double _value(ArrayDouble &coeffs,
                ulong start,
                ulong end);

  void _call(ArrayDouble &coeffs,
             double step,
             ArrayDouble &out,
             ulong start,
             ulong end);

  inline virtual void set_positive(bool positive) {
      this->positive = positive;
  }
};

#endif  // TICK_OPTIM_PROX_SRC_PROX_EQUALITY_H_
