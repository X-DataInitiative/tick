// License: BSD 3 clause

//
// Created by Martin Bompaire on 26/11/15.
//

#include "tick/base/base.h"
#include "tick/hawkes/simulation/hawkes_kernels/hawkes_kernel_sum_exp.h"

// By default, approximated fast formula for computing exponentials are not used
bool HawkesKernelSumExp::use_fast_exp = false;

void HawkesKernelSumExp::rewind() {
  last_convolution_values = ArrayDouble(n_decays);
  last_convolution_values.init_to_zero();
  convolution_restart_index = 0;
  last_convolution_time = 0;
  intensities_all_positive = (intensities.size() > 0 && intensities.min() >= 0);
}

HawkesKernelSumExp::HawkesKernelSumExp(const ArrayDouble &intensities,
                                       const ArrayDouble &decays)
  : HawkesKernel() {
  n_decays = decays.size();

  if (n_decays != intensities.size())
    throw std::invalid_argument("Intensities and decays arrays of HawkesKernelSumExp "
                                  "must have the same length");

  if (n_decays == 0)
    throw std::invalid_argument("Intensities and decays arrays of HawkesKernelSumExp "
                                  "must contain at least one value");

  support = std::numeric_limits<double>::max();
  this->intensities = intensities;
  this->decays = decays;

  if (decays.size() > 0 && decays.min() < 0)
    throw std::invalid_argument("All decays of HawkesKernelSumExp must be positive");

  rewind();
}

HawkesKernelSumExp::HawkesKernelSumExp(const HawkesKernelSumExp &kernel)
  : HawkesKernel(kernel) {
  n_decays = kernel.n_decays;
  intensities = kernel.intensities;
  decays = kernel.decays;
  rewind();
}

HawkesKernelSumExp::HawkesKernelSumExp()
  : HawkesKernelSumExp(ArrayDouble{1}, ArrayDouble{1}) {
}

// kernel value for one part of the sum
double HawkesKernelSumExp::get_value_i(double x, ulong i) {
  if (intensities[i] == 0) return 0;
  return intensities[i] * decays[i] * cexp(-decays[i] * x);
}

// The kernel values
double HawkesKernelSumExp::get_value_(double x) {
  double value = 0;

  for (ulong i = 0; i < n_decays; ++i) {
    value += get_value_i(x, i);
  }

  return value;
}

// Compute the convolution kernel*process(time)
double HawkesKernelSumExp::get_convolution(const double time, const ArrayDouble &timestamps,
                                           double *const bound) {
  if (timestamps.size() < convolution_restart_index) {
    throw std::runtime_error("HawkesKernelSumExp cannot get convolution on an "
                               "another process unless it has been rewound");
  }
  double delay = time - last_convolution_time;
  if (delay < 0) {
    throw std::runtime_error("HawkesKernelSumExp cannot get convolution on an "
                               "older time unless it has been rewound");
  }

  double value{0.};
  for (ulong i = 0; i < n_decays; ++i) {
    if (delay > 0) {
      last_convolution_values[i] *= cexp(-decays[i] * delay);
    }
  }

  ulong k;
  for (k = convolution_restart_index; k < timestamps.size(); ++k) {
    double t_k = timestamps[k];
    if (t_k > time) break;
    for (ulong i = 0; i < n_decays; ++i) {
      last_convolution_values[i] += get_value_i(time - t_k, i);
    }
  }

  last_convolution_time = time;
  convolution_restart_index = k;

  value = last_convolution_values.sum();

  if (bound) {
    if (!intensities_all_positive) {
      *bound = 0;
      for (ulong u = 0; u < n_decays; ++u) {
        if (intensities[u] > 0) *bound += last_convolution_values[u];
      }
    } else {
      *bound = value;
    }
  }

  return value;
}

double HawkesKernelSumExp::get_norm(int nsteps) {
  return intensities.sum();
}

SArrayDoublePtr HawkesKernelSumExp::get_intensities() {
  ArrayDouble intensities_copy = intensities;
  return intensities_copy.as_sarray_ptr();
}

SArrayDoublePtr HawkesKernelSumExp::get_decays() {
  ArrayDouble decays_copy = decays;
  return decays_copy.as_sarray_ptr();
}
