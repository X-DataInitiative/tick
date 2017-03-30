//
// Created by Stéphane GAIFFAS on 30/12/2015.
//

#ifndef TICK_OPTIM_PROX_SRC_PROX_POSITIVE_H_
#define TICK_OPTIM_PROX_SRC_PROX_POSITIVE_H_

#include "prox_separable.h"

class ProxPositive : public ProxSeparable {
 public:
    explicit ProxPositive(double strength);

    ProxPositive(double strength, ulong start, ulong end);

    const std::string get_class_name() const;

    virtual void _call_i(ulong i,
                         ArrayDouble &coeffs,
                         double step,
                         ArrayDouble &out) const;

    double _value(ArrayDouble &coeffs,
                  ulong start,
                  ulong end);
};

#endif  // TICK_OPTIM_PROX_SRC_PROX_POSITIVE_H_
