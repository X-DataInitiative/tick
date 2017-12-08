// License: BSD 3 clause

#include "tick/prox/prox_l2.h"

ProxL2::ProxL2(double strength,
                   bool positive)
  : Prox(strength, positive) {}

ProxL2::ProxL2(double strength,
                   ulong start,
                   ulong end,
                   bool positive)
  : Prox(strength, start, end, positive) {}

const std::string ProxL2::get_class_name() const {
  return "ProxL2";
}


void ProxL2::call(const ArrayDouble &coeffs,
                  double step,
                  ArrayDouble &out,
                  ulong start,
                  ulong end) {
  ArrayDouble sub_coeffs = view(coeffs, start, end);
  ArrayDouble sub_out = view(out, start, end);
  const double thresh = step * strength * std::sqrt(end - start);
  double norm = std::sqrt(sub_coeffs.norm_sq());

  if (norm <= thresh) {
    sub_out.fill(0.);
  } else {
    double t = 1. - thresh / norm;
    sub_out *= t;
  }
  if (positive) {
    for (ulong i = 0; i < sub_out.size(); ++i) {
      if (sub_out[i] < 0) {
        sub_out[i] = 0;
      }
    }
  }
}


double ProxL2::value(const ArrayDouble &coeffs,
                     ulong start,
                     ulong end) {
  double norm_sq = view(coeffs, start, end).norm_sq();
  return strength * std::sqrt((end - start) * norm_sq);
}
