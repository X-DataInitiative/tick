// License: BSD 3 clause


#include "tick/hawkes/simulation/hawkes_kernels/hawkes_kernel_power_law.h"

HawkesKernelPowerLaw::HawkesKernelPowerLaw(double multiplier,
                                           double cutoff,
                                           double exponent,
                                           double support,
                                           double error)
  : HawkesKernel(support),
    multiplier(multiplier), exponent(exponent), cutoff(cutoff) {
  if (support <= 0) {
    if (error <= 0) {
      throw std::invalid_argument("Either support or error must be non negative");
    }
    this->support = pow(error, -1 / exponent) / multiplier - cutoff;
  }
}

HawkesKernelPowerLaw::HawkesKernelPowerLaw()
  : HawkesKernelPowerLaw(0.0, 0.0, 0.0) {
}

double HawkesKernelPowerLaw::get_value_(double x) {
  return multiplier * pow(x + cutoff, -exponent);
}

double HawkesKernelPowerLaw::get_norm(int nsteps) {
  double A = support;
  double norm =
    (pow(A + cutoff, 1 - exponent) - pow(cutoff, 1 - exponent)) * multiplier / (1 - exponent);
  return norm;
}

