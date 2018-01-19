// License: BSD 3 clause

#include "tick/prox/prox_l2sq.h"

ProxL2Sq::ProxL2Sq(double strength,
                   bool positive)
  : ProxSeparable(strength, positive) {}

ProxL2Sq::ProxL2Sq(double strength,
                   ulong start,
                   ulong end,
                   bool positive)
  : ProxSeparable(strength, start, end, positive) {}

const std::string ProxL2Sq::get_class_name() const {
  return "ProxL2Sq";
}

// Compute the prox on the i-th coordinate only
double ProxL2Sq::call_single(double x,
                             double step) const {
  if (positive && x < 0) {
    return 0;
  } else {
    return x / (1 + step * strength);
  }
}

// Repeat n_times the prox on coordinate i
double ProxL2Sq::call_single(double x,
                             double step,
                             ulong n_times) const {
  if (n_times >= 1) {
    if (positive && x < 0) {
      return 0;
    } else {
      return x / std::pow(1 + step * strength, n_times);
    }
  } else {
    return x;
  }
}

double ProxL2Sq::value_single(double x) const {
  return x * x / 2;
}
