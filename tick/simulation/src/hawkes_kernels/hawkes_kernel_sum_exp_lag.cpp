// License: BSD 3 clause

//
// Created by Martin Bompaire on 26/11/15.
//

#include "base.h"
#include "hawkes_kernel_sum_exp_lag.h"

// By default, approximated fast formula for computing exponentials are not used
bool HawkesKernelSumExpLag::use_fast_exp = false;

void HawkesKernelSumExpLag::rewind() {
  last_convolution_values = ArrayDouble(n_decays);
  last_convolution_values.init_to_zero();
    convolution_restart_indexs = ArrayULong(n_decays);
    convolution_restart_indexs.init_to_zero();
  last_convolution_time = 0;
  intensities_all_positive = (intensities.size() > 0 && intensities.min() >= 0);
}

HawkesKernelSumExpLag::HawkesKernelSumExpLag(const ArrayDouble &intensities,
                                       const ArrayDouble &decays, ArrayDouble lags)
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
  this->lags = lags;

  if (decays.size() > 0 && decays.min() < 0)
    throw std::invalid_argument("All decays of HawkesKernelSumExp must be positive");

  rewind();
}

HawkesKernelSumExpLag::HawkesKernelSumExpLag(const HawkesKernelSumExpLag &kernel)
    : HawkesKernel(kernel) {
  n_decays = kernel.n_decays;
  intensities = kernel.intensities;
  decays = kernel.decays;
  lags = kernel.lags;
    rewind();
}

HawkesKernelSumExpLag::HawkesKernelSumExpLag()
    : HawkesKernelSumExpLag(ArrayDouble{1}, ArrayDouble{1}, ArrayDouble{1}) {
  printf("Default constructor called.\n");
}

// kernel value for one part of the sum
double HawkesKernelSumExpLag::get_value_i(double x, ulong i) {
  if (intensities[i] == 0) return 0;
  if (x < lags[i]) return 0;
  return intensities[i] * decays[i] * cexp(-decays[i] * (x - lags[i]));
}

// The kernel values
double HawkesKernelSumExpLag::get_value_(double x) {
  double value = 0;

  for (ulong i = 0; i < n_decays; ++i) {
    value += get_value_i(x, i);
  }

  return value;
}

// Compute the convolution kernel*process(time)
double HawkesKernelSumExpLag::get_convolution(const double time, const ArrayDouble &timestamps,
                                           double *const bound) {
    if (timestamps.size() < convolution_restart_indexs.max()) {
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

  for (ulong i = 0; i < n_decays; ++i) {
    for (ulong k = convolution_restart_indexs[i]; k < timestamps.size(); ++k) {
      double t_k = timestamps[k];
      if (t_k > time - lags[i]) break;
      last_convolution_values[i] += get_value_i(time - t_k, i);
      convolution_restart_indexs[i] = k + 1;
    }
  }

  last_convolution_time = time;

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

double HawkesKernelSumExpLag::get_norm(int nsteps) {
  return intensities.sum();
}

SArrayDoublePtr HawkesKernelSumExpLag::get_intensities() {
  ArrayDouble intensities_copy = intensities;
  return intensities_copy.as_sarray_ptr();
}

SArrayDoublePtr HawkesKernelSumExpLag::get_decays() {
  ArrayDouble decays_copy = decays;
  return decays_copy.as_sarray_ptr();
}
