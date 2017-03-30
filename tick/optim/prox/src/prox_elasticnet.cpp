//
// Created by Maryan Morel on 08/03/16.
//

#include "prox_elasticnet.h"

ProxElasticNet::ProxElasticNet(double strength,
                               double ratio,
                               bool positive)
    : ProxSeparable(strength) {
    if (ratio < 0 || ratio > 1)
        TICK_ERROR("Ratio should be in the [0, 1] interval");

    this->positive = positive;
    this->ratio = ratio;
}

ProxElasticNet::ProxElasticNet(double strength,
                               double ratio,
                               ulong start,
                               ulong end,
                               bool positive)
    : ProxSeparable(strength, start, end) {
    if (ratio < 0 || ratio > 1)
        TICK_ERROR("Ratio should be in the [0, 1] interval");

    this->positive = positive;
    this->ratio = ratio;
}

const std::string ProxElasticNet::get_class_name() const {
    return "ProxElasticNet";
}

double ProxElasticNet::_value_i(ulong i, ArrayDouble &coeffs) const {
    double coeffs_i = coeffs[i];
    double value = (1 - ratio) * 0.5 * coeffs_i * coeffs_i;
    if (coeffs_i > 0) {
        value += ratio * coeffs_i;
    } else {
        value -= ratio * coeffs_i;
    }
    return value;
}

void ProxElasticNet::_call_i(ulong i,
                             ArrayDouble &coeffs,
                             double step,
                             ArrayDouble &out) const {
    double thresh = step * ratio * strength;
    double coeffs_i = coeffs[i];
    if (coeffs_i > 0) {
        if (coeffs_i > thresh) {
            out[i] = (coeffs_i - thresh) / (1 + step * strength * (1 - ratio));
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
                out[i] = (coeffs_i + thresh) / (1 + step * strength * (1 - ratio));
            } else {
                out[i] = 0;
            }
        }
    }
}
