// License: BSD 3 clause


#include "hawkes_kernel_exp_lag.h"

// By default, approximated fast formula for computing exponentials are not used
bool HawkesKernelExpLag::use_fast_exp = false;

void HawkesKernelExpLag::rewind() {
  last_convolution_time = 0;
  convolution_restart_index = 0;
  last_convolution_value = 0;
}

// constructor
HawkesKernelExpLag::HawkesKernelExpLag(double intensity, double decay, double lag)
    : intensity(intensity), decay(decay), lag(lag) {
  if (decay < 0)
    throw std::invalid_argument("Decay of HawkesKernelExpLag must be positive");
  support = std::numeric_limits<double>::max();
  rewind();
}

// copy constructor
HawkesKernelExpLag::HawkesKernelExpLag(const HawkesKernelExpLag &kernel)
    : HawkesKernel(kernel) {
  intensity = kernel.intensity;
  decay = kernel.decay;
  lag = kernel.lag;
  rewind();
}

HawkesKernelExpLag::HawkesKernelExpLag()
    : HawkesKernelExpLag(0.0, 0.0, 0.0) {
}

// Getting the value of the kernel at the point x
double HawkesKernelExpLag::get_value_(double x) {
  if (intensity == 0)
    return 0;

  if (x < lag)
      return 0;

  return intensity * decay * cexp(-decay * (x - lag));
}

double HawkesKernelExpLag::get_norm(int nsteps) {
  double norm = intensity;
  return norm;
}

// Returns the convolution kernel*process(time)
double HawkesKernelExpLag::get_convolution(const double time,
                                        const ArrayDouble &timestamps,
                                        double *const bound) {
  double value{0.};
  if (intensity == 0 || time < 0) {
    // value stays at 0
  } else {
    if (timestamps.size() < convolution_restart_index) {
      throw std::runtime_error("HawkesKernelExpLag cannot get convolution on an "
                                   "another process unless it has been rewound");
    }
    double delay = time - last_convolution_time;
    if (delay < 0) {
      throw std::runtime_error("HawkesKernelExpLag cannot get convolution on an "
                                   "older time unless it has been rewound");
    }

    value = last_convolution_value * cexp(-decay * delay);

    ulong k;
    for (k = convolution_restart_index; k < timestamps.size(); ++k) {
      double t_k = timestamps[k];
      if (t_k > time -lag) break;
      value += get_value(time - t_k);
      convolution_restart_index = k + 1;
    }

    last_convolution_time = time;
    last_convolution_value = value;
  }

  if (bound) {
    if (intensity >= 0) {
      // kernel is decreasing
      *bound = value;
    } else {
      // kernel is increasing
      *bound = 0;
    }
  }

  return value;
}

