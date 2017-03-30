//
// Created by Stéphane GAIFFAS on 30/12/2015.
//

#ifndef TICK_OPTIM_PROX_SRC_PROX_L1W_H_
#define TICK_OPTIM_PROX_SRC_PROX_L1W_H_

#include "base.h"
#include "prox_separable.h"

class ProxL1w : public ProxSeparable {
 protected:
    bool positive;

    // Weights for L1 penalization
    SArrayDoublePtr weights;

 public:
    ProxL1w(double strength,
            SArrayDoublePtr weights,
            bool positive);

    ProxL1w(double strength,
            SArrayDoublePtr weights,
            ulong start, ulong end, bool positive);

    const std::string get_class_name() const;

    virtual double _value_i(ulong i,
                            ArrayDouble &coeffs) const;

    virtual void _call_i(ulong i,
                         ArrayDouble &coeffs,
                         double step,
                         ArrayDouble &out) const;

    inline virtual void set_weights(SArrayDoublePtr weights) {
        this->weights = weights;
    }

    inline virtual void set_positive(bool positive) {
        this->positive = positive;
    }
};

#endif  // TICK_OPTIM_PROX_SRC_PROX_L1W_H_
