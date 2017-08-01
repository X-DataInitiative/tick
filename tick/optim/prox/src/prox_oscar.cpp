// License: BSD 3 clause

#include "prox_oscar.h"

ProxOscar::ProxOscar(double strength,
                     double ratio,
                     bool positive)
  : ProxSortedL1(strength, positive) {
  set_ratio(ratio);
}

ProxOscar::ProxOscar(double strength,
                     double ratio,
                     ulong start,
                     ulong end,
                     bool positive)
  : ProxSortedL1(strength, start, end, positive) {
  set_ratio(ratio);
}

const std::string ProxOscar::get_class_name() const {
  return "ProxOscar";
}

void ProxOscar::compute_weights(void) {
  if (!weights_ready) {
    ulong size = end - start;
    weights = ArrayDouble(size);
    for (ulong i = 0; i < size; i++) {
      weights[i] = strength * (ratio * (size - i - 1)  + 1);
    }
    weights_ready = true;
  }
}

double ProxOscar::get_ratio() const {
  return ratio;
}

void ProxOscar::set_ratio(double ratio) {
  if (ratio < 0) {
    TICK_ERROR("Ratio should be non-negative");
  } else {
    if (ratio != this->ratio) {
      weights_ready = false;
      this->ratio = ratio;
    }
  }
}
