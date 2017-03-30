//
// Created by Martin Bompaire on 26/10/15.
//

#ifndef TICK_OPTIM_PROX_SRC_PROX_L2SQ_H_
#define TICK_OPTIM_PROX_SRC_PROX_L2SQ_H_

#include "prox_separable.h"

class ProxL2Sq : public ProxSeparable {
 protected:
    bool positive;

 public:
    ProxL2Sq(double strength, bool positive);

    ProxL2Sq(double strength, ulong start, ulong end, bool positive);

    const std::string get_class_name() const;

    inline virtual void set_positive(bool positive) {
        this->positive = positive;
    }

    virtual double _value_i(ulong i,
                            ArrayDouble &coeffs) const;

    virtual void _call_i(ulong i,
                         ArrayDouble &coeffs,
                         double step,
                         ArrayDouble &out) const;
};

#endif  // TICK_OPTIM_PROX_SRC_PROX_L2SQ_H_
