// License: BSD 3 clause

#include "tick/hawkes/simulation/hawkes_kernels/hawkes_kernel_time_func.h"

HawkesKernelTimeFunc::HawkesKernelTimeFunc(const TimeFunction &time_function)
  : HawkesKernel(), time_function(time_function) {
  if (time_function.get_border_type() != TimeFunction::BorderType::Border0) TICK_ERROR(
    "Only TimeFunction with a border 0 can be used in HawkesKernelTimeFunc");

  support = time_function.get_support_right();
}

HawkesKernelTimeFunc::HawkesKernelTimeFunc(const ArrayDouble &t_axis, const ArrayDouble &y_axis)
  : HawkesKernelTimeFunc(TimeFunction(t_axis, y_axis)) {
}

HawkesKernelTimeFunc::HawkesKernelTimeFunc()
  : HawkesKernelTimeFunc(TimeFunction(0.0)) {
}

double HawkesKernelTimeFunc::get_value_(double x) {
  return time_function.value(x);
}

double HawkesKernelTimeFunc::get_future_max(double t, double value_at_t) {
  return time_function.future_bound(t);
}


