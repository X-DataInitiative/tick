//
// Created by Martin Bompaire on 04/03/16.
//

#include "prox_separable.h"

ProxSeparable::ProxSeparable(double strength)
    : Prox(strength) {}

ProxSeparable::ProxSeparable(double strength,
                             ulong start,
                             ulong end)
    : Prox(strength, start, end) {}

double ProxSeparable::_value(ArrayDouble &coeffs,
                             ulong start,
                             ulong end) {
    double value = 0;
    // We work on a view, so that sub_coeffs and weights are "aligned"
    // (namely both ranging between 0 and end - start).
    // This is particularly convenient for Prox classes with weights for each
    // coordinate
    ArrayDouble sub_coeffs = view(coeffs, start, end);
    for (ulong i = 0; i < end - start; ++i) {
        value += _value_i(i, sub_coeffs);
    }
    return strength * value;
}

void ProxSeparable::call(ArrayDouble &coeffs,
                         ArrayDouble &step,
                         ArrayDouble &out) {
    if (has_range) {
        if (end > coeffs.size())
            TICK_ERROR("Range [" << start << ", " << end << "] cannot be called on a vector of size " << coeffs.size());

        if (step.size() != end - start)
            TICK_ERROR("step must be of size " << end - start);

        _call(coeffs, step, out, start, end);
    } else {
        if (step.size() != coeffs.size())
            TICK_ERROR("step must have the same size as coeffs ");

        _call(coeffs, step, out, 0, coeffs.size());
    }
}

void ProxSeparable::_call(ArrayDouble &coeffs,
                          double step,
                          ArrayDouble &out,
                          ulong start,
                          ulong end) {
    ArrayDouble sub_coeffs = view(coeffs, start, end);
    ArrayDouble sub_out = view(out, start, end);
    for (ulong i = 0; i < end - start; ++i) {
        // Call the prox on each coordinate
        _call_i(i, sub_coeffs, step, sub_out);
    }
}

void ProxSeparable::_call(ArrayDouble &coeffs,
                          ArrayDouble &step,
                          ArrayDouble &out,
                          ulong start,
                          ulong end) {
    ArrayDouble sub_coeffs = view(coeffs, start, end);
    ArrayDouble sub_out = view(out, start, end);
    for (ulong i = 0; i < end - start; ++i) {
        _call_i(i, sub_coeffs, step[i], sub_out);
    }
}

double ProxSeparable::_value_i(ulong i,
                               ArrayDouble &coeffs) const {
    TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

// Compute the prox on the i-th coordinate only
void ProxSeparable::_call_i(ulong i,
                            ArrayDouble &coeffs,
                            double t,
                            ArrayDouble &out) const {
    TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}
