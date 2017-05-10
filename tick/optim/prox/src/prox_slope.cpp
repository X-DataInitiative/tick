
#include "prox_slope.h"


ProxSlope::ProxSlope(double strength,
                     double fdr,
                     bool positive)
  : ProxSortedL1(strength, WeightsType::bh, positive) {
  this->fdr = fdr;
}

ProxSlope::ProxSlope(double strength,
                     double fdr,
                     ulong start,
                     ulong end,
                     bool positive)
  : ProxSortedL1(strength, WeightsType::bh, start, end, positive) {
  this->fdr = fdr;
}

const std::string ProxSlope::get_class_name() const {
  return "ProxSlope";
}

void ProxSlope::compute_weights(void) {
  if (!weights_ready) {
    ulong size = end - start;
    weights = ArrayDouble(size);
    for (ulong i = 0; i < size; i++) {
      double tmp = fdr / (2 * size);
      weights[i] = strength * standard_normal_inv_cdf(1 - tmp * (i + 1));
    }
    weights_ready = true;
  }
}
