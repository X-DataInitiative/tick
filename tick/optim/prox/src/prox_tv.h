//
// Created by Stéphane GAIFFAS on 30/12/2015.
//

#ifndef TICK_OPTIM_PROX_SRC_PROX_TV_H_
#define TICK_OPTIM_PROX_SRC_PROX_TV_H_

#include "prox.h"

class ProxTV : public Prox {
 protected:
    bool positive;

 public:
    ProxTV(double strength, bool positive);

    ProxTV(double strength, ulong start, ulong end,
           bool positive);

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

#endif  // TICK_OPTIM_PROX_SRC_PROX_TV_H_
