// License: BSD 3 clause

#include "tick/hawkes/simulation/hawkes_kernels/hawkes_kernel.h"

// Constructor
HawkesKernel::HawkesKernel(double support) : support(support) {}

// Copy constructor
HawkesKernel::HawkesKernel(const HawkesKernel &kernel) { support = kernel.support; }

// The main method to get kernel values
double HawkesKernel::get_value(double x) { return ((x >= support || x < 0) ? 0 : get_value_(x)); }

// The main method to get the values of the primitive of the kernel
double HawkesKernel::get_primitive_value(double t) {
  if (support <= 0) return 0;
  return ((t <= 0) ? 0 : get_primitive_value_(t));
}
// The main method to get the values of the primitive from s to t
double HawkesKernel::get_primitive_value(double s, double t) {
  if (support <= 0) return 0;
  if (s <= 0) s = 0;
  if (t < s) {
    throw std::runtime_error("HawkesKernel cannot compute the primitive from s to t if s>t");
  }
  return get_primitive_value(t - s);
}

// Get a shared array representing the kernel values on the t_values
SArrayDoublePtr HawkesKernel::get_values(const ArrayDouble &t_values) {
  SArrayDoublePtr y_values = SArrayDouble::new_ptr(t_values.size());
  for (ulong i = 0; i < y_values->size(); ++i) {
    (*y_values)[i] = get_value(t_values[i]);
  }
  return y_values;
}

// Get L1 norm
// By default, it discretizes the integral with nsteps (Riemann sum with
// step-wise function) Should be overloaded if L1 norm closed formula exists
double HawkesKernel::get_norm(int nsteps) {
  double norm = 0;
  double dx = support / nsteps;

  if (support == 0) return 0;

  for (double x = 0; x <= support; x += dx) norm += get_value(x) * dx;

  return norm;
}

// Returns the convolution kernel*process(time)
// If bound != NULL then *bound will return a bound of the future values of the
// convolution, i.e., MAX(kernel*process(t>=time)) Should be overloaded for
// efficiency if there is a faster way to compute this convolution than just
// regular algorithm (e.g. for exponential kernels).
double HawkesKernel::get_convolution(const double time, const ArrayDouble &timestamps,
                                     double *const bound) {
  if (bound) *bound = 0;
  if (is_zero()) return 0;

  double value = 0;
  ulong k = timestamps.size();
  double firstTime = time - get_support();

  while (k >= 1 && timestamps[k - 1] >= firstTime) {
    double t = timestamps[k - 1];
    double new_event_value = get_value(time - t);
    value += new_event_value;
    if (bound) {
      *bound += get_future_max(time - t, new_event_value);
    }
    k--;
  }

  return value;
}

// Returns the convolution of the process with the primitive of the kernel.
// Should be overloaded for efficiency if there is a faster way
// to compute this convolution than just regular algorithm (e.g. for exponential kernels).
double HawkesKernel::get_primitive_convolution(const double t, const ArrayDouble &timestamps) {
  if (is_zero()) return 0;
  double value = 0;
  ulong n = timestamps.size();
  for (ulong k = 0; k < n; ++k) {
    double t_k = timestamps[k];
    if (t_k >= t) break;
    value += get_primitive_value(t_k, t);
  }
  return value;
}
