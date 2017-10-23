//
// Created by Maryan Morel on 01/08/2017.
//

#ifndef TICK_OPTIM_PROX_SRC_ProxL1L2_H_
#define TICK_OPTIM_PROX_SRC_ProxL1L2_H_

#include "prox.h"

class ProxL1L2 : public Prox {
 public:
  ProxL1L2(double strength, bool positive);

  // TODO: add weights + find a better API (right now, need ProxMulti to make groups)
  ProxL1L2(double strength, ulong start, ulong end, bool positive);

  const std::string get_class_name() const override;

  double value(const ArrayDouble &coeffs, ulong start, ulong end) override;

  void call(const ArrayDouble &coeffs, double step, ArrayDouble &out,
            ulong start, ulong end) override;
};

#endif //TICK_OPTIM_PROX_SRC_ProxL1L2_H_
