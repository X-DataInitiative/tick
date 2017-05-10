
#include "base.h"
#include "prox_equality.h"

ProxEquality::ProxEquality(double strength, bool positive)
  : Prox(0.) {
  this->positive = positive;
}

ProxEquality::ProxEquality(double strength,
                           ulong start,
                           ulong end,
                           bool positive)
  : Prox(0., start, end) {
  this->positive = positive;
}

const std::string ProxEquality::get_class_name() const {
  return "ProxEquality";
}

double ProxEquality::_value(ArrayDouble &coeffs,
                            ulong start,
                            ulong end) {
  return 0;
}

void ProxEquality::_call(ArrayDouble &coeffs,
                         double step,
                         ArrayDouble &out,
                         ulong start,
                         ulong end) {
  ArrayDouble sub_coeffs = view(coeffs, start, end);
  ArrayDouble sub_out = view(out, start, end);
  double mean = sub_coeffs.sum() / sub_coeffs.size();
  sub_out.fill(mean);
  if (positive && (mean < 0)) {
    sub_out.fill(0);
  }
}
