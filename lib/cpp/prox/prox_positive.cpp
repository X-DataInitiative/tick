// License: BSD 3 clause

#include "tick/prox/prox_positive.h"

ProxPositive::ProxPositive(double strength)
  : TProxSeparable<double, double>(strength, true) {}

ProxPositive::ProxPositive(double strength,
                           ulong start,
                           ulong end)
  : TProxSeparable<double, double>(strength, start, end, true) {}

std::string ProxPositive::get_class_name() const {
  return "ProxPositive";
}

double ProxPositive::call_single(double x,
                                 double step) const {
  if (x < 0) {
    return 0;
  } else {
    return x;
  }
}

double ProxPositive::call_single(double x,
                                 double step,
                                 ulong n_times) const {
  return call_single(x, step);
}

double ProxPositive::value(const ArrayDouble &coeffs,
                           ulong start,
                           ulong end) {
  return 0.;
}
