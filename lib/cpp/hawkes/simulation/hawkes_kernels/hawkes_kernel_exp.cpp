// License: BSD 3 clause

#include <cassert>

#include "tick/hawkes/simulation/hawkes_kernels/hawkes_kernel_exp.h"

// By default, approximated fast formula for computing exponentials are not used
bool HawkesKernelExp::use_fast_exp = false;

void HawkesKernelExp::rewind() {
  last_convolution_time = 0;
  convolution_restart_index = 0;
  last_convolution_value = 0;
  last_primitive_convolution_time = 0;
  primitive_convolution_restart_index = 0;
  last_primitive_convolution_value = 0;
}

// constructor
HawkesKernelExp::HawkesKernelExp(double intensity, double decay)
    : intensity(intensity), decay(decay) {
  if (decay < 0) throw std::invalid_argument("Decay of HawkesKernelExp must be positive");
  support = std::numeric_limits<double>::max();
  rewind();
}

// copy constructor
HawkesKernelExp::HawkesKernelExp(const HawkesKernelExp &kernel) : HawkesKernel(kernel) {
  intensity = kernel.intensity;
  decay = kernel.decay;
  rewind();
}

HawkesKernelExp::HawkesKernelExp() : HawkesKernelExp(0.0, 0.0) {}

// Getting the value of the kernel at the point x
double HawkesKernelExp::get_value_(double x) {
  if (intensity == 0) return 0;

  return intensity * decay * cexp(-decay * x);
}

double HawkesKernelExp::get_norm(int nsteps) {
  double norm = intensity;
  return norm;
}

// Getting the value of the primitive of the kernel at points s<t
double HawkesKernelExp::get_primitive_value_(double t) {
  if (intensity == 0) return 0;
  assert(t >= 0);
  return intensity - intensity * exp(-decay * t);
}

// Returns the convolution kernel*process(time)
double HawkesKernelExp::get_convolution(const double time, const ArrayDouble &timestamps,
                                        double *const bound) {
  double value{0.};
  if (intensity == 0 || time < 0) {
    // value stays at 0
  } else {
    if (timestamps.size() < convolution_restart_index) {
      throw std::runtime_error(
          "HawkesKernelExp cannot get convolution on an "
          "another process unless it has been rewound");
    }
    double delay = time - last_convolution_time;
    if (delay < 0) {
      throw std::runtime_error(
          "HawkesKernelExp cannot get convolution on an "
          "older time unless it has been rewound");
    }

    value = last_convolution_value * cexp(-decay * delay);

    ulong k;
    for (k = convolution_restart_index; k < timestamps.size(); ++k) {
      double t_k = timestamps[k];
      if (t_k > time) break;
      value += get_value(time - t_k);
    }

    last_convolution_time = time;
    last_convolution_value = value;
    convolution_restart_index = k;
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

// Returns the convolution of the process with primitive of the kernel
double HawkesKernelExp::get_primitive_convolution(const double time,
                                                  const ArrayDouble &timestamps) {
  double value{0.};
  if (intensity == 0 || time < 0) {
    // value stays at 0
  } else {
    ulong n = timestamps.size();
    ulong m = primitive_convolution_restart_index;
    double a = intensity;
    double b = decay;

    if (n < m) {
      throw std::runtime_error(
          "HawkesKernelExp cannot get convolution on an "
          "another process unless it has been rewound");
    }
    double delay = time - last_primitive_convolution_time;
    if (delay < 0) {
      throw std::runtime_error(
          "HawkesKernelExp cannot get convolution on an "
          "older time unless it has been rewound");
    }
    if (m == 0) {
      value = 0.;
    } else {
      value = a * m - ((a * m - last_primitive_convolution_value) * cexp(-b * delay));
    }

    ulong k;
    for (k = m; k < n; ++k) {
      double t_k = timestamps[k];
      if (t_k >= time) break;
      value += get_primitive_value(t_k, time);
    }
    if (value < last_primitive_convolution_value) {
      throw std::runtime_error("last_primitive_convolution_value > new value");
    }

    last_primitive_convolution_time = time;
    last_primitive_convolution_value = value;
    primitive_convolution_restart_index = k;
  }
  return value;
}
