//
// Created by Stéphane GAIFFAS on 30/12/2015.
//

#include "prox_l1w.h"

ProxL1w::ProxL1w(double strength,
                 SArrayDoublePtr weights,
                 bool positive)
    : ProxSeparable(strength) {
    this->positive = positive;
    this->weights = weights;
}

ProxL1w::ProxL1w(double strength,
                 SArrayDoublePtr weights,
                 ulong start,
                 ulong end,
                 bool positive)
    : ProxSeparable(strength, start, end) {
    this->positive = positive;
    this->weights = weights;
}

const std::string ProxL1w::get_class_name() const {
    return "ProxL1w";
}

double ProxL1w::_value_i(ulong i,
                         ArrayDouble &coeffs) const {
    double coeffs_i = coeffs[i];
    if (coeffs_i > 0) {
        return coeffs_i * (*weights)[i];
    } else {
        return -coeffs_i * (*weights)[i];
    }
}

void ProxL1w::_call_i(ulong i,
                      ArrayDouble &coeffs,
                      double step,
                      ArrayDouble &out) const {
    double thresh = step * strength;
    double coeffs_i = coeffs[i];
    double thresh_i = thresh * (*weights)[i];
    if (coeffs_i > 0) {
        if (coeffs_i > thresh_i) {
            out[i] = coeffs_i - thresh_i;
        } else {
            out[i] = 0;
        }
    } else {
        // If coeffs_i is negative and we project onto the non-negative half-plane
        // we set it to 0
        if (positive) {
            out[i] = 0;
        } else {
            if (coeffs_i < -thresh_i) {
                out[i] = coeffs_i + thresh_i;
            } else {
                out[i] = 0;
            }
        }
    }
}
