//
// Created by Maryan Morel on 01/08/2017.
//

#include "prox_l1l2.h"
#include <math.h>

ProxL1L2::ProxL1L2(double strength,
                   bool positive)
    : Prox(strength, positive) {}

ProxL1L2::ProxL1L2(double strength,
                   ulong start,
                   ulong end,
                   bool positive)
    : Prox(strength, start, end, positive) {}

const std::string ProxL1L2::get_class_name() const {
  return "ProxL1L2";
}

void ProxL1L2::call(const ArrayDouble &coeffs,
                    double step,
                    ArrayDouble &out,
                    ulong start,
                    ulong end) {
  double norm = this->value(coeffs, start, end) / strength;
  // TODO: dumb to compute `mult` each time as groups are define outside this
  // prox with ProxMulti for now
  // + might be improved with if(norm <= strength){out = 0}else{out = mult * coeffs}
  double mult = 1 - strength / norm;
  mult = (mult<0)?0:mult;

  for(ulong i(start); i < end; i++){
    out[i] = mult * coeffs[i];
  }
}

double ProxL1L2::value(const ArrayDouble &coeffs,
                       ulong start,
                       ulong end) {
  ArrayDouble group_coeffs = view(coeffs, start, end);
  double group_norm_sq = group_coeffs.norm_sq();
  return strength * sqrt(group_norm_sq);
}