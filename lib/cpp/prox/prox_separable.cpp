// License: BSD 3 clause

#include "tick/prox/prox_separable.h"

ProxSeparable::ProxSeparable(double strength, bool positive)
    : Prox(strength, positive) {}

ProxSeparable::ProxSeparable(double strength, ulong start, ulong end, bool positive)
    : Prox(strength, start, end, positive) {}

const std::string ProxSeparable::get_class_name() const {
  return "ProxSeparable";
}

const bool ProxSeparable::is_separable() const {
  return true;
}

void ProxSeparable::call(const ArrayDouble &coeffs,
                         const ArrayDouble &step,
                         ArrayDouble &out) {
  if (has_range) {
    if (end > coeffs.size()) TICK_ERROR(
        "Range [" << start << ", " << end
                  << "] cannot be called on a vector of size " << coeffs.size());
    if (step.size() != end - start) TICK_ERROR("step must be of size " << end - start);

    call(coeffs, step, out, start, end);
  } else {
    if (step.size() != coeffs.size()) TICK_ERROR("step must have the same size as coeffs ");
    call(coeffs, step, out, 0, coeffs.size());
  }
}

void ProxSeparable::call(const ArrayDouble &coeffs,
                         double step,
                         ArrayDouble &out,
                         ulong start,
                         ulong end) {
  ArrayDouble sub_coeffs = view(coeffs, start, end);
  ArrayDouble sub_out = view(out, start, end);
  for (ulong i = 0; i < sub_coeffs.size(); ++i) {
    // Call the prox on each coordinate
    sub_out[i] = call_single(sub_coeffs[i], step);
  }
}

void ProxSeparable::call(const ArrayDouble &coeffs,
                         const ArrayDouble &step,
                         ArrayDouble &out,
                         ulong start,
                         ulong end) {
  ArrayDouble sub_coeffs = view(coeffs, start, end);
  ArrayDouble sub_out = view(out, start, end);
  for (ulong i = 0; i < sub_coeffs.size(); ++i) {
    sub_out[i] = call_single(sub_coeffs[i], step[i]);
  }
}

double ProxSeparable::call_single(double x,
                                  double step) const {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

double ProxSeparable::call_single(double x,
                                  double step,
                                  ulong n_times) const {
  if (n_times >= 1) {
    for (ulong r = 0; r < n_times; ++r) {
      x = call_single(x, step);
    }
  }
  return x;
}

// Compute the prox on the i-th coordinate only
void ProxSeparable::call_single(ulong i,
                                const ArrayDouble &coeffs,
                                double step,
                                ArrayDouble &out) const {
  if (i >= coeffs.size()) {
    TICK_ERROR(get_class_name() << "::call_single " << "i= " << i << " while coeffs.size()=" << coeffs.size());
  } else {
    if (has_range) {
      if ((i >= start) && (i < end)) {
        out[i] = call_single(coeffs[i], step);
      } else {
        out[i] = coeffs[i];
      }
    } else {
      out[i] = call_single(coeffs[i], step);
    }
  }
}

// Repeat n_times the prox on coordinate i
void ProxSeparable::call_single(ulong i,
                                const ArrayDouble &coeffs,
                                double step,
                                ArrayDouble &out,
                                ulong n_times) const {
  if (i >= coeffs.size()) {
    TICK_ERROR(get_class_name() << "::call_single " << "i= " << i << " while coeffs.size()=" << coeffs.size());
  } else {
    if (has_range) {
      if ((i >= start) && (i < end)) {
        out[i] = call_single(coeffs[i], step, n_times);
      } else {
        out[i] = coeffs[i];
      }
    } else {
      out[i] = call_single(coeffs[i], step, n_times);
    }
  }
}

double ProxSeparable::value(const ArrayDouble &coeffs,
                            ulong start,
                            ulong end) {
  double val = 0;
  // We work on a view, so that sub_coeffs and weights are "aligned"
  // (namely both ranging between 0 and end - start).
  // This is particularly convenient for Prox classes with weights for each
  // coordinate
  ArrayDouble sub_coeffs = view(coeffs, start, end);
  for (ulong i = 0; i < sub_coeffs.size(); ++i) {
    val += value_single(sub_coeffs[i]);
  }
  return strength * val;
}

double ProxSeparable::value_single(double x) const {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}
