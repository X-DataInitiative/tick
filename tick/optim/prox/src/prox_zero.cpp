//
// Created by Stéphane GAIFFAS on 29/12/2015.
//

#include "prox_zero.h"

ProxZero::ProxZero(double strength)
    : ProxSeparable(strength) {}

ProxZero::ProxZero(double strength,
                   ulong start,
                   ulong end)
    : ProxSeparable(strength, start, end) {}

const std::string ProxZero::get_class_name() const {
    return "ProxZero";
}

double ProxZero::_value(ArrayDouble &coeffs,
                        ulong start,
                        ulong end) {
    return 0.;
}

void ProxZero::_call(ArrayDouble &coeffs,
                     double step,
                     ArrayDouble &out,
                     ulong start,
                     ulong end) {
    // We copy the contents of coeffs into out
    ArrayDouble sub_coeffs = view(coeffs, start, end);
    ArrayDouble sub_out = view(out, start, end);
    for (unsigned int i = 0; i < sub_coeffs.size(); ++i) {
        sub_out[i] = sub_coeffs[i];
    }
}

void ProxZero::_call(ArrayDouble &coeffs,
                     ArrayDouble &step,
                     ArrayDouble &out,
                     ulong start,
                     ulong end) {
    // We copy the contents of coeffs into out
    ArrayDouble sub_coeffs = view(coeffs, start, end);
    ArrayDouble sub_out = view(out, start, end);
    for (unsigned int i = 0; i < sub_coeffs.size(); ++i) {
        sub_out[i] = sub_coeffs[i];
    }
}
