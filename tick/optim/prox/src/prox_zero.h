//
// Created by Stéphane GAIFFAS on 29/12/2015.
//

#ifndef TICK_OPTIM_PROX_SRC_PROX_ZERO_H_
#define TICK_OPTIM_PROX_SRC_PROX_ZERO_H_

#include "prox_separable.h"

class ProxZero : public ProxSeparable {
 public:
    explicit ProxZero(double strength);

    ProxZero(double strength, ulong start, ulong end);

    const std::string get_class_name() const;

    double _value(ArrayDouble &coeffs,
                  ulong start,
                  ulong end);

    virtual void _call(ArrayDouble &coeffs,
                       double step,
                       ArrayDouble &out,
                       ulong start,
                       ulong end);

    virtual void _call(ArrayDouble &coeffs,
                       ArrayDouble &step,
                       ArrayDouble &out,
                       ulong start,
                       ulong end);
};

#endif  // TICK_OPTIM_PROX_SRC_PROX_ZERO_H_
