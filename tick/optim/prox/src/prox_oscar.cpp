// License: BSD 3 clause

#include "prox_oscar.h"

ProxOscar::ProxOscar(double strength,
                     double ratio,
                     bool positive)
  : ProxSortedL1(strength, WeightsType::bh, positive) {
  this->ratio = ratio;
}

ProxOscar::ProxOscar(double strength,
                     double ratio,
                     ulong start,
                     ulong end,
                     bool positive)
  : ProxSortedL1(strength, WeightsType::bh, start, end, positive) {
  this->ratio = ratio;
}

const std::string ProxOscar::get_class_name() const {
  return "ProxOscar";
}

void ProxOscar::compute_weights(void) {
  if (!weights_ready) {
    ulong size = end - start;
    weights = ArrayDouble(size);
    for (ulong i = 0; i < size; i++) {
      // double tmp = false_discovery_rate / (2 * size);
      // weights[i] = strength * standard_normal_inv_cdf(1 - tmp * (i + 1));
      weights[i] = 0;
    }
    weights_ready = true;
  }
}

double ProxOscar::get_ratio() const {
  return ratio;
}

void ProxOscar::set_ratio(double ratio) {
  this->ratio = ratio;
}
