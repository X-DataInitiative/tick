// License: BSD 3 clause

#include "tick/prox/prox_separable.h"


template <class T, class K>
TProxSeparable<T, K>::TProxSeparable(K strength, bool positive)
    : TProx<T, K>(strength, positive) {}

template <class T, class K>
TProxSeparable<T, K>::TProxSeparable(K strength, ulong start, ulong end, bool positive)
    : TProx<T, K>(strength, start, end, positive) {}

template <class T, class K>
std::string
TProxSeparable<T, K>::get_class_name() const {
  return "ProxSeparable";
}

template <class T, class K>
bool
TProxSeparable<T, K>::is_separable() const {
  return true;
}

template <class T, class K>
void
TProxSeparable<T, K>::call(const Array<T> &coeffs,
                         const Array<K> &step,
                         Array<T> &out) {
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

template <class T, class K>
void
TProxSeparable<T, K>::call(const Array<T> &coeffs,
                         K step,
                         Array<T> &out,
                         ulong start,
                         ulong end) {
  Array<T> sub_coeffs = view(coeffs, start, end);
  Array<T> sub_out = view(out, start, end);
  for (ulong i = 0; i < sub_coeffs.size(); ++i) {
    // Call the prox on each coordinate
    sub_out[i] = call_single(sub_coeffs[i], step);
  }
}

template <class T, class K>
void
TProxSeparable<T, K>::call(const Array<T> &coeffs,
                         const Array<K> &step,
                         Array<T> &out,
                         ulong start,
                         ulong end) {
  Array<T> sub_coeffs = view(coeffs, start, end);
  Array<T> sub_out = view(out, start, end);
  for (ulong i = 0; i < sub_coeffs.size(); ++i) {
    sub_out[i] = call_single(sub_coeffs[i], step[i]);
  }
}

template <class T, class K>
K
TProxSeparable<T, K>::call_single(K x,
                                  K step) const {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

template <class T, class K>
K
TProxSeparable<T, K>::call_single(K x,
                                  K step,
                                  ulong n_times) const {
  if (n_times >= 1) {
    for (ulong r = 0; r < n_times; ++r) {
      x = call_single(x, step);
    }
  }
  return x;
}

// Compute the prox on the i-th coordinate only
template <class T, class K>
void
TProxSeparable<T, K>::call_single(ulong i,
                                const Array<T> &coeffs,
                                K step,
                                Array<T> &out) const {
  if (i >= coeffs.size()) {
    TICK_ERROR(get_class_name() << "::call_single " << "i= " << i << " while coeffs.size()=" << coeffs.size());
  } else {
    if (has_range) {
      if ((i >= start) && (i < end)) {
        set_out_i(out, i, call_single(coeffs[i], step));
      } else {
        set_out_i(out, i, coeffs[i]);
      }
    } else {
      set_out_i(out, i, call_single(coeffs[i], step));
    }
  }
}

// Repeat n_times the prox on coordinate i
template <class T, class K>
void
TProxSeparable<T, K>::call_single(
  ulong i,
  const Array<T> &coeffs,
  K step,
  Array<T> &out,
  ulong n_times
) const {
  if (i >= coeffs.size()) {
    TICK_ERROR(get_class_name() << "::call_single " << "i= " << i << " while coeffs.size()=" << coeffs.size());
  } else {
    if (has_range) {
      if ((i >= start) && (i < end)) {
        set_out_i(out, i, call_single(coeffs[i], step, n_times));
      } else {
        set_out_i(out, i, coeffs[i]);
      }
    } else {
      set_out_i(out, i, call_single(coeffs[i], step, n_times));
    }
  }
}

template <class T, class K>
K
TProxSeparable<T, K>::value(const Array<T> &coeffs,
                            ulong start,
                            ulong end) {
  K val = 0;
  // We work on a view, so that sub_coeffs and weights are "aligned"
  // (namely both ranging between 0 and end - start).
  // This is particularly convenient for Prox classes with weights for each
  // coordinate
  Array<T> sub_coeffs = view(coeffs, start, end);
  for (ulong i = 0; i < sub_coeffs.size(); ++i) {
    val += value_single(sub_coeffs[i]);
  }
  return strength * val;
}

template <class T, class K>
K
TProxSeparable<T, K>::value_single(K x) const {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

template <class T, class K>
void
TProxSeparable<T, K>::set_out_i(Array<T> &out, size_t i, K d) const {
  out[i] = d;
}

template class TProxSeparable<double, double>;
template class TProxSeparable<float , float >;
