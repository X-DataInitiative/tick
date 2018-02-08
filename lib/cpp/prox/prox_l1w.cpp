// License: BSD 3 clause

#include "tick/prox/prox_l1w.h"

template <class T>
TProxL1w<T>::TProxL1w(T strength, SArrayTPtr weights, bool positive)
    : TProxSeparable<T>(strength, positive) {
  this->weights = weights;
}

template <class T>
TProxL1w<T>::TProxL1w(T strength, SArrayTPtr weights, ulong start, ulong end,
                      bool positive)
    : TProxSeparable<T>(strength, start, end, positive) {
  this->weights = weights;
}

template <class T>
std::string TProxL1w<T>::get_class_name() const {
  return "TProxL1w";
}

template <class T>
T TProxL1w<T>::call_single(T x, T step) const {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

template <class T>
T TProxL1w<T>::call_single(T x, T step, ulong n_times) const {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

template <class T>
T TProxL1w<T>::value_single(T x) const {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

template <class T>
T TProxL1w<T>::call_single(T x, T step, T weight) const {
  T thresh = step * strength * weight;
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

template <class T>
T TProxL1w<T>::call_single(T x, T step, T weight, ulong n_times) const {
  if (n_times >= 1) {
    return call_single(x, n_times * step, weight);
  } else {
    return x;
  }
}

template <class T>
void TProxL1w<T>::call(const Array<T> &coeffs, T step, Array<T> &out,
                       ulong start, ulong end) {
  Array<T> sub_coeffs = view(coeffs, start, end);
  Array<T> sub_out = view(out, start, end);
  for (ulong i = 0; i < sub_coeffs.size(); ++i) {
    sub_out[i] = call_single(sub_coeffs[i], step, (*weights)[i]);
  }
}

template <class T>
void TProxL1w<T>::call(const Array<T> &coeffs, const Array<T> &step,
                       Array<T> &out, ulong start, ulong end) {
  Array<T> sub_coeffs = view(coeffs, start, end);
  Array<T> sub_out = view(out, start, end);
  for (ulong i = 0; i < sub_coeffs.size(); ++i) {
    // weights has the same size as end - start, but not the step array
    sub_out[i] = call_single(sub_coeffs[i], step[i + start], (*weights)[i]);
  }
}

// We cannot implement only TProxL1w<T>::call_single(T x, T step) since we need
// to know i to find the weight
template <class T>
void TProxL1w<T>::call_single(ulong i, const Array<T> &coeffs, T step,
                              Array<T> &out) const {
  if (i >= coeffs.size()) {
    TICK_ERROR(get_class_name()
               << "::call_single "
               << "i= " << i << " while coeffs.size()=" << coeffs.size());
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

template <class T>
void TProxL1w<T>::call_single(ulong i, const Array<T> &coeffs, T step,
                              Array<T> &out, ulong n_times) const {
  if (i >= coeffs.size()) {
    TICK_ERROR(get_class_name()
               << "::call_single "
               << "i= " << i << " while coeffs.size()=" << coeffs.size());
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

template <class T>
T TProxL1w<T>::value_single(T x, T weight) const {
  return weight * std::abs(x);
}

template <class T>
T TProxL1w<T>::value(const Array<T> &coeffs, ulong start, ulong end) {
  T val = 0;
  // We work on a view, so that sub_coeffs and weights are "aligned"
  // (namely both ranging between 0 and end - start).
  Array<T> sub_coeffs = view(coeffs, start, end);
  for (ulong i = 0; i < sub_coeffs.size(); ++i) {
    val += value_single(sub_coeffs[i], (*weights)[i]);
  }
  return strength * val;
}

template class TProxL1w<double>;
template class TProxL1w<float>;
