// License: BSD 3 clause

#include "tick/hawkes/simulation/hawkes_baselines/constant_baseline.h"

HawkesConstantBaseline::HawkesConstantBaseline(double value) : value(value) {}

double HawkesConstantBaseline::get_value(double t) { return value; }

double HawkesConstantBaseline::get_primitive_value(double t) {
  if (t <= 0) return 0;
  return t * value;
}

SArrayDoublePtr HawkesConstantBaseline::get_value(ArrayDouble &t) {
  SArrayDoublePtr values = SArrayDouble::new_ptr(t.size());
  values->fill(value);
  return values;
}

double HawkesConstantBaseline::get_future_bound(double t) { return value; }
