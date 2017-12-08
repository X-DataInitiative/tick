// License: BSD 3 clause

#include "tick/base/base.h"
#include "tick/prox/prox_equality.h"

ProxEquality::ProxEquality(double strength, bool positive)
  : Prox(0., positive) {}

ProxEquality::ProxEquality(double strength,
                           ulong start,
                           ulong end,
                           bool positive)
  : Prox(0., start, end, positive) {}

const std::string ProxEquality::get_class_name() const {
  return "ProxEquality";
}

double ProxEquality::value(const ArrayDouble &coeffs, ulong start, ulong end) {
  ArrayDouble sub_coeffs = view(coeffs, start, end);
  if (sub_coeffs.min() == sub_coeffs.max()) {
    return 0;
  } else {
    return std::numeric_limits<double>::max();
  }
}

void ProxEquality::call(const ArrayDouble &coeffs,
                        double step,
                        ArrayDouble &out,
                        ulong start,
                        ulong end) {
  ArrayDouble sub_coeffs = view(coeffs, start, end);
  ArrayDouble sub_out = view(out, start, end);
  double mean = sub_coeffs.sum() / sub_coeffs.size();
  if (positive && (mean < 0)) {
    sub_out.fill(0);
  } else {
    sub_out.fill(mean);
  }
}
