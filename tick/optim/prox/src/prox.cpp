//
// Created by Martin Bompaire on 26/10/15.
//

#include "prox.h"

Prox::Prox(double strength) {
    has_range = false;
    this->strength = strength;
}

Prox::Prox(double strength,
           ulong start,
           ulong end) {
    set_start_end(start, end);
    this->strength = strength;

    if (start > end)
        TICK_ERROR(get_class_name() << " can't have start(" << start << ") greater than end(" << end << ")");
}

const std::string Prox::get_class_name() const {
    return "Prox";
}

double Prox::value(ArrayDouble &coeffs) {
    if (has_range) {
        if (end > coeffs.size())
            TICK_ERROR(get_class_name() << " of range [" << start << ", " << end << "] cannot get value of a vector of size " << coeffs.size());

    } else {
        // If no range is given, we use the size of coeffs to get it
        set_start_end(0, coeffs.size());
        // But no range was given (set_start_end set has_range to true)
        has_range = false;
    }
    return _value(coeffs, start, end);
}

double Prox::_value(ArrayDouble &coeffs,
                    ulong start,
                    ulong end) {
    TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

void Prox::call(ArrayDouble &coeffs,
                double step,
                ArrayDouble &out) {
    if (has_range) {
        if (end > coeffs.size())
            TICK_ERROR(get_class_name() << " of range [" << start << ", " << end << "] cannot be called on a vector of size " << coeffs.size());

    } else {
        // If no range is given, we use the size of coeffs to get it
        set_start_end(0, coeffs.size());
        // But no range was given (set_start_end set has_range to true)
        has_range = false;
    }
    _call(coeffs, step, out, start, end);
}

void Prox::_call(ArrayDouble &coeffs,
                 double step,
                 ArrayDouble &out,
                 ulong start,
                 ulong end) {
    TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

void Prox::call(ArrayDouble &coeffs,
                ArrayDouble &step,
                ArrayDouble &out) {
    // This is overloaded in ProxSeparable.
    // If the child class does not inherit from ProxSeparable, it cannot use this method.
    TICK_WARNING() << "Method not implemented since this prox is not separable.";
    TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

void Prox::set_strength(double strength) {
    this->strength = strength;
}

double Prox::get_strength() const {
    return strength;
}

void Prox::set_start_end(ulong start, ulong end) {
    this->has_range = true;
    this->start = start;
    this->end = end;
}

ulong Prox::get_start() {
    return start;
}

ulong Prox::get_end() {
    return end;
}
