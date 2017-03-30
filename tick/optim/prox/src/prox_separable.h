//
// Created by Martin Bompaire on 04/03/16.
//

#ifndef TICK_OPTIM_PROX_SRC_PROX_SEPARABLE_H_
#define TICK_OPTIM_PROX_SRC_PROX_SEPARABLE_H_

#include "prox.h"

class ProxSeparable : public Prox {
 public:
    explicit ProxSeparable(double strength);

    ProxSeparable(double strength,
                  ulong start,
                  ulong end);

    virtual double _value(ArrayDouble &coeffs,
                          ulong start,
                          ulong end);

    virtual void _call(ArrayDouble &coeffs,
                       double step,
                       ArrayDouble &out,
                       ulong start,
                       ulong end);

    virtual void call(ArrayDouble &coeffs,
                      ArrayDouble &step,
                      ArrayDouble &out);

    virtual void _call(ArrayDouble &coeffs,
                       ArrayDouble &step,
                       ArrayDouble &out,
                       ulong start,
                       ulong end);

    // Compute the value given by the i-th coordinate only (multiplication by lambda must
    // not be done here)
    virtual double _value_i(ulong i,
                            ArrayDouble &coeffs) const;

    // Compute the prox on the i-th coordinate only
    virtual void _call_i(ulong i,
                         ArrayDouble &coeffs,
                         double step,
                         ArrayDouble &out) const;
};

#endif  // TICK_OPTIM_PROX_SRC_PROX_SEPARABLE_H_
