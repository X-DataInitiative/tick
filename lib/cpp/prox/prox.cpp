// License: BSD 3 clause

#include "tick/prox/prox.h"

template <class T, class K>
TProx<T, K>::TProx(
  K strength,
  bool positive
) {
  has_range = false;
  this->strength = strength;
  this->positive = positive;
}

template <class T, class K>
TProx<T, K>::TProx(
  K strength,
  ulong start,
  ulong end,
  bool positive
) : TProx<T, K>(strength, positive) {
  set_start_end(start, end);
}

template <class T, class K>
std::string
TProx<T, K>::get_class_name() const {
  return "TProx";
}

template <class T, class K>
bool
TProx<T, K>::is_separable() const {
  return false;
}

template <class T, class K>
void
TProx<T, K>::call(const Array<T> &coeffs,
                K step,
                Array<T> &out) {
  if (has_range) {
    if (end > coeffs.size()) TICK_ERROR(
      get_class_name() << " of range [" << start << ", " << end
                       << "] cannot be called on a vector of size " << coeffs.size());
  } else {
    // If no range is given, we use the size of coeffs to get it
    set_start_end(0, coeffs.size());
    // But no range was given (set_start_end set has_range to true)
    // Still it is mandatory to set `start` and `end` has some prox might need to know if
    // they have changed from one iteration to another (see ProxSortedL1)
    has_range = false;
  }
  call(coeffs, step, out, start, end);
}

template <class T, class K>
void
TProx<T, K>::call(const Array<T> &coeffs,
                K step,
                Array<T> &out,
                ulong start,
                ulong end) {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

template <class T, class K>
K
TProx<T, K>::value(const Array<T> &coeffs) {
  if (has_range) {
    if (end > coeffs.size()) TICK_ERROR(
      get_class_name() << " of range [" << start << ", " << end
                       << "] cannot get value of a vector of size " << coeffs.size());
  } else {
    // If no range is given, we use the size of coeffs to get it
    set_start_end(0, coeffs.size());
    // But no range was given (set_start_end set has_range to true)
    has_range = false;
  }
  return value(coeffs, start, end);
}

template <class T, class K>
K
TProx<T, K>::value(
  const Array<T> &coeffs,
  ulong start,
  ulong end) {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

template <class T, class K>
K
TProx<T, K>::get_strength() const {
  return strength;
}

template <class T, class K>
void
TProx<T, K>::set_strength(K strength) {
  this->strength = strength;
}

template <class T, class K>
void
TProx<T, K>::set_start_end(ulong start,
                         ulong end) {
  if (start >= end)
    TICK_ERROR(
      get_class_name() << " can't have start(" << start
                       << ") greater or equal than end("
                       << end << ")");
  this->has_range = true;
  this->start = start;
  this->end = end;
}

template <class T, class K>
ulong
TProx<T, K>::get_start() const {
  return start;
}

template <class T, class K>
ulong
TProx<T, K>::get_end() const {
  return end;
}

template <class T, class K>
bool
TProx<T, K>::get_positive() const {
  return positive;
}

template <class T, class K>
void
TProx<T, K>::set_positive(bool positive) {
  this->positive = positive;
}

template class TProx<double, double>;
template class TProx<float , float>;
