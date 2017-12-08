// License: BSD 3 clause

#include "tick/prox/prox_zero.h"

ProxZero::ProxZero(double strength)
  : ProxSeparable(strength, false) {}

ProxZero::ProxZero(double strength,
                   ulong start,
                   ulong end)
  : ProxSeparable(strength, start, end, false) {}

const std::string ProxZero::get_class_name() const {
  return "ProxZero";
}

double ProxZero::call_single(double x,
                             double step) const {
  return x;
}

double ProxZero::call_single(double x,
                             double step,
                             ulong n_times) const {
  return x;
}

double ProxZero::value(const ArrayDouble &coeffs,
                       ulong start,
                       ulong end) {
  return 0.;
}
