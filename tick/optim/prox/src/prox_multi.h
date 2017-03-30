//
// Created by Stéphane GAIFFAS on 30/12/2015.
//

#ifndef TICK_OPTIM_PROX_SRC_PROX_MULTI_H_
#define TICK_OPTIM_PROX_SRC_PROX_MULTI_H_

#include "prox.h"


class ProxMulti : public Prox {
 protected:
    std::vector<ProxPtr> proxs;

 public:
    explicit ProxMulti(std::vector<ProxPtr> proxs, double strength);

    const std::string get_class_name() const;

    double _value(ArrayDouble &coeffs,
                  ulong start,
                  ulong end);

    void _call(ArrayDouble &coeffs,
               double step,
               ArrayDouble &out,
               ulong start,
               ulong end);
};

#endif  // TICK_OPTIM_PROX_SRC_PROX_MULTI_H_
