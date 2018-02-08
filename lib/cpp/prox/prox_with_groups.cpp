// License: BSD 3 clause

#include "tick/prox/prox_with_groups.h"

template <class T>
TProxWithGroups<T>::TProxWithGroups(T strength, SArrayULongPtr blocks_start,
                                    SArrayULongPtr blocks_length, bool positive)
    : TProx<T>(strength, positive), is_synchronized(false) {
  this->blocks_start = blocks_start;
  this->blocks_length = blocks_length;
  this->positive = positive;
  // blocks_start and blocks_end have the same size
  n_blocks = blocks_start->size();
  // The object is not ready (synchronize_proxs has not been called)
  is_synchronized = false;
}

template <class T>
TProxWithGroups<T>::TProxWithGroups(T strength, SArrayULongPtr blocks_start,
                                    SArrayULongPtr blocks_length, ulong start,
                                    ulong end, bool positive)
    : TProx<T>(strength, start, end, positive) {
  this->blocks_start = blocks_start;
  this->blocks_length = blocks_length;
  this->positive = positive;
  // blocks_start and blocks_end have the same size
  n_blocks = blocks_start->size();
  // The object is not ready (synchronize_proxs has not been called)
  is_synchronized = false;
}

template <class T>
void TProxWithGroups<T>::synchronize_proxs() {
  proxs.clear();
  for (ulong k = 0; k < n_blocks; k++) {
    ulong start = (*blocks_start)[k];
    if (has_range) {
      // If there is a range, we apply the global start
      start += this->start;
    }
    ulong end = start + (*blocks_length)[k];
    proxs.emplace_back(build_prox(strength, start, end, positive));
  }
  is_synchronized = true;
}

template <class T>
std::unique_ptr<TProx<T> > TProxWithGroups<T>::build_prox(T strength,
                                                          ulong start,
                                                          ulong end,
                                                          bool positive) {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

template <class T>
std::string TProxWithGroups<T>::get_class_name() const {
  return "TProxWithGroups";
}

template <class T>
T TProxWithGroups<T>::value(const Array<T> &coeffs, ulong start, ulong end) {
  if (!is_synchronized) {
    synchronize_proxs();
  }
  T val = 0.;
  for (auto &prox : proxs) {
    val += prox->value(coeffs, prox->get_start(), prox->get_end());
  }
  return val;
}

template <class T>
void TProxWithGroups<T>::call(const Array<T> &coeffs, T step, Array<T> &out,
                              ulong start, ulong end) {
  if (!is_synchronized) {
    synchronize_proxs();
  }
  for (auto &prox : proxs) {
    prox->call(coeffs, step, out, prox->get_start(), prox->get_end());
  }
}

template class TProxWithGroups<double>;
template class TProxWithGroups<float>;
