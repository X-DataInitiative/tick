// License: BSD 3 clause

#include "tick/prox/prox_l1w.h"

ProxL1w::ProxL1w(double strength,
                 SArrayDoublePtr weights,
                 bool positive)
    : ProxSeparable(strength, positive) {
  this->weights = weights;
}

ProxL1w::ProxL1w(double strength,
                 SArrayDoublePtr weights,
                 ulong start,
                 ulong end,
                 bool positive)
    : ProxSeparable(strength, start, end, positive) {
  this->weights = weights;
}

const std::string ProxL1w::get_class_name() const {
  return "ProxL1w";
}

double ProxL1w::call_single(double x, double step) const {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

double ProxL1w::call_single(double x, double step, ulong n_times) const {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

double ProxL1w::value_single(double x) const {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

double ProxL1w::call_single(double x, double step, double weight) const {
  double thresh = step * strength * weight;
  if (x > 0) {
    if (x > thresh) {
      return x - thresh;
    } else {
      return 0;
    }
  } else {
    // If coeffs_i is negative we set it to 0
    if (positive) {
      return 0;
    } else {
      if (x < -thresh) {
        return x + thresh;
      } else {
        return 0;
      }
    }
  }
}

double ProxL1w::call_single(double x, double step, double weight, ulong n_times) const {
  if (n_times >= 1) {
    return call_single(x, n_times * step, weight);
  } else {
    return x;
  }
}

void ProxL1w::call(const ArrayDouble &coeffs,
                   double step,
                   ArrayDouble &out,
                   ulong start,
                   ulong end) {
  ArrayDouble sub_coeffs = view(coeffs, start, end);
  ArrayDouble sub_out = view(out, start, end);
  for (ulong i = 0; i < sub_coeffs.size(); ++i) {
    sub_out[i] = call_single(sub_coeffs[i], step, (*weights)[i]);
  }
}

void ProxL1w::call(const ArrayDouble &coeffs,
                   const ArrayDouble &step,
                   ArrayDouble &out,
                   ulong start,
                   ulong end) {
  ArrayDouble sub_coeffs = view(coeffs, start, end);
  ArrayDouble sub_out = view(out, start, end);
  for (ulong i = 0; i < sub_coeffs.size(); ++i) {
    // weights has the same size as end - start, but not the step array
    sub_out[i] = call_single(sub_coeffs[i], step[i + start], (*weights)[i]);
  }
}

// We cannot implement only ProxL1w::call_single(double x, double step) since we need to
// know i to find the weight
void ProxL1w::call_single(ulong i,
                          const ArrayDouble &coeffs,
                          double step,
                          ArrayDouble &out) const {
  if (i >= coeffs.size()) {
    TICK_ERROR(get_class_name() << "::call_single " << "i= " << i
                                << " while coeffs.size()=" << coeffs.size());
  } else {
    if (has_range) {
      if ((i >= start) && (i < end)) {
        out[i] = call_single(coeffs[i], step, (*weights)[i - start]);
      } else {
        out[i] = coeffs[i];
      }
    } else {
      out[i] = call_single(coeffs[i], step, (*weights)[i - start]);
    }
  }
}

void ProxL1w::call_single(ulong i,
                          const ArrayDouble &coeffs,
                          double step,
                          ArrayDouble &out,
                          ulong n_times) const {
  if (i >= coeffs.size()) {
    TICK_ERROR(get_class_name() << "::call_single " << "i= " << i
                                << " while coeffs.size()=" << coeffs.size());
  } else {
    if (has_range) {
      if ((i >= start) && (i < end)) {
        out[i] = call_single(coeffs[i], step, (*weights)[i - start], n_times);
      } else {
        out[i] = coeffs[i];
      }
    } else {
      out[i] = call_single(coeffs[i], step, (*weights)[i - start], n_times);
    }
  }
}

double ProxL1w::value_single(double x, double weight) const {
  return weight * std::abs(x);
}

double ProxL1w::value(const ArrayDouble &coeffs, ulong start, ulong end) {
  double val = 0;
  // We work on a view, so that sub_coeffs and weights are "aligned"
  // (namely both ranging between 0 and end - start).
  ArrayDouble sub_coeffs = view(coeffs, start, end);
  for (ulong i = 0; i < sub_coeffs.size(); ++i) {
    val += value_single(sub_coeffs[i], (*weights)[i]);
  }
  return strength * val;
}
