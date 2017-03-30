//
// Created by Stéphane GAIFFAS on 30/12/2015.
//

#ifndef TICK_OPTIM_PROX_SRC_PROX_L1_H_
#define TICK_OPTIM_PROX_SRC_PROX_L1_H_

#include "prox_separable.h"

class ProxL1 : public ProxSeparable {
 protected:
    bool positive;

 public:
    ProxL1(double strength, bool positive);

    ProxL1(double strength, ulong start, ulong end, bool positive);

    const std::string get_class_name() const;

    virtual double _value_i(ulong i,
                            ArrayDouble &coeffs) const;

    virtual void _call_i(ulong i,
                         ArrayDouble &coeffs,
                         double step,
                         ArrayDouble &out) const;

    inline virtual void set_positive(bool positive) {
        this->positive = positive;
    }
};

#endif  // TICK_OPTIM_PROX_SRC_PROX_L1_H_
