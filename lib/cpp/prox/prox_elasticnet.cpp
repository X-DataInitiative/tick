// License: BSD 3 clause

#include "tick/prox/prox_elasticnet.h"

ProxElasticNet::ProxElasticNet(double strength,
                               double ratio,
                               bool positive)
  : ProxSeparable(strength, positive) {
  this->positive = positive;
  set_ratio(ratio);
}

ProxElasticNet::ProxElasticNet(double strength,
                               double ratio,
                               ulong start,
                               ulong end,
                               bool positive)
  : ProxSeparable(strength, start, end, positive) {
  this->positive = positive;
  set_ratio(ratio);
}

const std::string ProxElasticNet::get_class_name() const {
  return "ProxElasticNet";
}

double ProxElasticNet::call_single(double x,
                                   double step) const {
  double thresh = step * ratio * strength;
  if (x > 0) {
    if (x > thresh) {
      return (x - thresh) / (1 + step * strength * (1 - ratio));
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
        return (x + thresh) / (1 + step * strength * (1 - ratio));
      } else {
        return 0;
      }
    }
  }
}

double ProxElasticNet::value_single(double x) const {
  return (1 - ratio) * 0.5 * x * x + ratio * std::abs(x);
}

double ProxElasticNet::get_ratio() const {
  return ratio;
}

void ProxElasticNet::set_ratio(double ratio) {
  if (ratio < 0 || ratio > 1) TICK_ERROR("Ratio should be in the [0, 1] interval");
  this->ratio = ratio;
}
