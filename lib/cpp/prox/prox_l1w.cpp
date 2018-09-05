// License: BSD 3 clause

#include "tick/prox/prox_l1w.h"

template <class T, class K>
T TProxL1w<T, K>::call_single(T x, T step) const {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

template <class T, class K>
T TProxL1w<T, K>::call_single(T x, T step, ulong n_times) const {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

template <class T, class K>
T TProxL1w<T, K>::value_single(T x) const {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

template <class T, class K>
T TProxL1w<T, K>::call_single(T x, T step, T weight) const {
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

template <class T, class K>
T TProxL1w<T, K>::call_single(T x, T step, T weight, ulong n_times) const {
  if (n_times >= 1) {
    return call_single(x, n_times * step, weight);
  } else {
    return x;
  }
}

template <class T, class K>
void TProxL1w<T, K>::call(const Array<K> &coeffs, T step, Array<K> &out,
                          ulong start, ulong end) {
  auto sub_coeffs = view(coeffs, start, end);
  auto sub_out = view(out, start, end);
  for (ulong i = 0; i < sub_coeffs.size(); ++i) {
    sub_out[i] = call_single(sub_coeffs[i], step, (*weights)[i]);
  }
}

template <class T, class K>
void TProxL1w<T, K>::call(const Array<K> &coeffs, const Array<T> &step,
                          Array<K> &out, ulong start, ulong end) {
  auto sub_coeffs = view(coeffs, start, end);
  auto sub_out = view(out, start, end);
  for (ulong i = 0; i < sub_coeffs.size(); ++i) {
    // weights has the same size as end - start, but not the step array
    sub_out[i] = call_single(sub_coeffs[i], step[i + start], (*weights)[i]);
  }
}

template <class T, class K>
T TProxL1w<T, K>::call_single_with_index(T x, T step, ulong i) const {
  return is_in_range(i) ? call_single(x, step, (*weights)[i - start]) : x;
}

// We cannot implement only TProxL1w<T, K>::call_single(T x, T step) since we
// need to know i to find the weight
template <class T, class K>
void TProxL1w<T, K>::call_single(ulong i, const Array<K> &coeffs, T step,
                                 Array<K> &out) const {
  if (i >= coeffs.size()) {
    TICK_ERROR(get_class_name()
               << "::call_single "
               << "i= " << i << " while coeffs.size()=" << coeffs.size());
  } else {
    if (has_range) {
      if ((i >= start) && (i < end)) {
        out[i] = call_single(coeffs[i], step, (*weights)[i - start]);
      } else {
        out.set_data_index(i, coeffs[i]);
      }
    } else {
      out[i] = call_single(coeffs[i], step, (*weights)[i - start]);
    }
  }
}

template <class T, class K>
void TProxL1w<T, K>::call_single(ulong i, const Array<K> &coeffs, T step,
                                 Array<K> &out, ulong n_times) const {
  if (i >= coeffs.size()) {
    TICK_ERROR(get_class_name()
               << "::call_single "
               << "i= " << i << " while coeffs.size()=" << coeffs.size());
  } else {
    if (has_range) {
      if ((i >= start) && (i < end)) {
        out[i] = call_single(coeffs[i], step, (*weights)[i - start], n_times);
      } else {
        out.set_data_index(i, coeffs[i]);
      }
    } else {
      out[i] = call_single(coeffs[i], step, (*weights)[i - start], n_times);
    }
  }
}

template <class T, class K>
T TProxL1w<T, K>::value_single(T x, T weight) const {
  return weight * std::abs(x);
}

template <class T, class K>
T TProxL1w<T, K>::value(const Array<K> &coeffs, ulong start, ulong end) {
  T val = 0;
  // We work on a view, so that sub_coeffs and weights are "aligned"
  // (namely both ranging between 0 and end - start).
  auto sub_coeffs = view(coeffs, start, end);
  for (ulong i = 0; i < sub_coeffs.size(); ++i) {
    val += value_single(sub_coeffs[i], (*weights)[i]);
  }
  return strength * val;
}

template class DLL_PUBLIC TProxL1w<double, double>;
template class DLL_PUBLIC TProxL1w<float, float>;

template class DLL_PUBLIC TProxL1w<double, std::atomic<double>>;
template class DLL_PUBLIC TProxL1w<float, std::atomic<float>>;
