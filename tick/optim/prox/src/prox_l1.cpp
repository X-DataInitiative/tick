//
// Created by Stéphane GAIFFAS on 30/12/2015.
//

#include "prox_l1.h"

ProxL1::ProxL1(double strength,
               bool positive)
    : ProxSeparable(strength) {
    this->positive = positive;
}

ProxL1::ProxL1(double strength,
               ulong start,
               ulong end,
               bool positive)
    : ProxSeparable(strength, start, end) {
    this->positive = positive;
}

const std::string ProxL1::get_class_name() const {
    return "ProxL1";
}

double ProxL1::_value_i(ulong i,
                        ArrayDouble &coeffs) const {
    double coeffs_i = coeffs[i];
    if (coeffs_i > 0) {
        return coeffs_i;
    } else {
        return -coeffs_i;
    }
}

void ProxL1::_call_i(ulong i,
                     ArrayDouble &coeffs,
                     double step,
                     ArrayDouble &out) const {
    double thresh = step * strength;
    double coeffs_i = coeffs[i];
    if (coeffs_i > 0) {
        if (coeffs_i > thresh) {
            out[i] = coeffs_i - thresh;
        } else {
            out[i] = 0;
        }
    } else {
        // If coeffs_i is negative and we project onto the non-negative half-plane
        // we set it to 0
        if (positive) {
            out[i] = 0;
        } else {
            if (coeffs_i < -thresh) {
                out[i] = coeffs_i + thresh;
            } else {
                out[i] = 0;
            }
        }
    }
}

