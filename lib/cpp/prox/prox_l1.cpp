// License: BSD 3 clause

#include "tick/prox/prox_l1.h"

ProxL1::ProxL1(double strength,
               bool positive)
    : ProxSeparable(strength, positive) {}

ProxL1::ProxL1(double strength,
               ulong start,
               ulong end,
               bool positive)
    : ProxSeparable(strength, start, end, positive) {}

const std::string ProxL1::get_class_name() const {
  return "ProxL1";
}

double ProxL1::call_single(double x,
                           double step) const {
  double thresh = step * strength;
  if (x > 0) {
    if (x > thresh) {
      return x - thresh;
    } else {
      return 0;
    }
  } else {
    // If x is negative and we project onto the non-negative half-plane
    // we set it to 0
    if (positive) {
      return 0;
    } else {
      if (x < -thresh) {
        return x + thresh;
      } else {
        return 0;
      }
    }
  }
}

double ProxL1::call_single(double x,
                           double step,
                           ulong n_times) const {
  if (n_times >= 1) {
    return call_single(x, n_times * step);
  } else {
    return x;
  }
}

double ProxL1::value_single(double x) const {
  return std::abs(x);
}
