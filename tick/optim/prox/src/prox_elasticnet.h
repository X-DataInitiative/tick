//
// Created by Maryan Morel on 08/03/16.
//

#ifndef TICK_OPTIM_PROX_SRC_PROX_ELASTICNET_H_
#define TICK_OPTIM_PROX_SRC_PROX_ELASTICNET_H_

#include "prox_separable.h"

class ProxElasticNet : public ProxSeparable {
 protected:
    bool positive;
    double ratio;

 public:
    ProxElasticNet(double strength, double ratio, bool positive);

    ProxElasticNet(double strength, double ratio, ulong start, ulong end, bool positive);

    const std::string get_class_name() const;

    double _value_i(ulong i, ArrayDouble &coeffs) const;

    void _call_i(ulong i, ArrayDouble &coeffs, double step, ArrayDouble &out) const;

    inline virtual void set_positive(bool positive) {
        this->positive = positive;
    }

    inline virtual void set_ratio(double ratio) {
        if (ratio < 0 || ratio > 1)
            TICK_ERROR("Ratio should be in the [0, 1] interval");

        this->ratio = ratio;
    }

    inline virtual double get_ratio() const {
        return ratio;
    }
};

#endif  // TICK_OPTIM_PROX_SRC_PROX_ELASTICNET_H_
