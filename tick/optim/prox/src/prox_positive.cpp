//
// Created by Stéphane GAIFFAS on 30/12/2015.
//

#include "prox_positive.h"

ProxPositive::ProxPositive(double strength)

    : ProxSeparable(strength) {}

ProxPositive::ProxPositive(double strength,
                           ulong start,
                           ulong end)

    : ProxSeparable(strength, start, end) {}

const std::string ProxPositive::get_class_name() const {
    return "ProxPositive";
}

double ProxPositive::_value(ArrayDouble &coeffs,
                            ulong start,
                            ulong end) {
    return 0.;
}

void ProxPositive::_call_i(ulong i,
                           ArrayDouble &coeffs,
                           double step,
                           ArrayDouble &out) const {
    double coeffs_i = coeffs[i];
    if (coeffs_i < 0) {
        out[i] = 0;
    } else {
        out[i] = coeffs_i;
    }
}

