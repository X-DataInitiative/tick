//
// Created by Martin Bompaire on 26/10/15.
//

#include "prox_l2sq.h"

ProxL2Sq::ProxL2Sq(double strength,
                   bool positive)

    : ProxSeparable(strength) {
    this->positive = positive;
}

ProxL2Sq::ProxL2Sq(double strength,
                   ulong start,
                   ulong end,
                   bool positive)

    : ProxSeparable(strength, start, end) {
    this->positive = positive;
}

const std::string ProxL2Sq::get_class_name() const {
    return "ProxL2Sq";
}

double ProxL2Sq::_value_i(ulong i,
                          ArrayDouble &coeffs) const {
    double coeffs_i = coeffs[i];
    return 0.5 * coeffs_i * coeffs_i;
}

// Compute the prox on the i-th coordinate only
void ProxL2Sq::_call_i(ulong i,
                       ArrayDouble &coeffs,
                       double step,
                       ArrayDouble &out) const {
    double coeffs_i = coeffs[i];
    if (positive && coeffs_i < 0) {
        out[i] = 0;
    } else {
        out[i] = coeffs_i / (1 + step * strength);
    }
}


void ProxL2Sq::_call_i(ulong i,
                       ArrayDouble &coeffs,
                       double step,
                       ArrayDouble &out,
                       ulong repeat) const {
    double coeffs_i = coeffs[i];
    if (repeat <= 0) {
        return;
    }
    if (positive && coeffs_i < 0) {
        out[i] = 0;
    } else {
        double c = 1.0 / pow(1.0 + step * strength, repeat);
        out[i] = c * coeffs_i;
    }
}
