#include "prox_l1w.h"

ProxL1w::ProxL1w(double strength,
                 SArrayDoublePtr weights,
                 bool positive)
    : ProxSeparable(strength, positive) {
    this->weights = weights;
}

ProxL1w::ProxL1w(double strength,
                 SArrayDoublePtr weights,
                 ulong start,
                 ulong end,
                 bool positive)
    : ProxSeparable(strength, start, end, positive) {
    this->weights = weights;
}

const std::string ProxL1w::get_class_name() const {
    return "ProxL1w";
}

void ProxL1w::call_single(ulong i,
                          const ArrayDouble &coeffs,
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
        // If coeffs_i is negative we set it to 0
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

void ProxL1w::call_single(ulong i,
                          const ArrayDouble &coeffs,
                          double step,
                          ArrayDouble &out,
                          ulong n_times) const {
    if (n_times >= 1) {
        call_single(i, coeffs, n_times * step, out);
    } else {
        out[i] = coeffs[i];
    }
}

double ProxL1w::value_single(ulong i,
                             const ArrayDouble &coeffs) const {
    return (*weights)[i] * std::abs(coeffs[i]);
}
