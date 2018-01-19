// License: BSD 3 clause


#include "tick/hawkes/simulation/hawkes_baselines/timefunction_baseline.h"

HawkesTimeFunctionBaseline::HawkesTimeFunctionBaseline()
  : time_function(0.) {}

HawkesTimeFunctionBaseline::HawkesTimeFunctionBaseline(TimeFunction time_function)
  : time_function(time_function) {}

HawkesTimeFunctionBaseline::HawkesTimeFunctionBaseline(ArrayDouble &times,
                                                       ArrayDouble &values) {
  time_function = TimeFunction(times, values,
                               TimeFunction::BorderType::Cyclic,
                               TimeFunction::InterMode::InterConstRight);
}

double HawkesTimeFunctionBaseline::get_value(double t) {
  return time_function.value(t);
}

SArrayDoublePtr HawkesTimeFunctionBaseline::get_value(ArrayDouble &t) {
  return time_function.value(t);
}

double HawkesTimeFunctionBaseline::get_future_bound(double t) {
  return time_function.future_bound(t);
}
