// License: BSD 3 clause

#include "tick/prox/prox_with_groups.h"

template <class T, class K>
TProxWithGroups<T, K>::TProxWithGroups(K strength,
                               SArrayULongPtr blocks_start,
                               SArrayULongPtr blocks_length,
                               bool positive)
    : TProx<K, K>(strength, positive), is_synchronized(false) {
  this->blocks_start = blocks_start;
  this->blocks_length = blocks_length;
  this->positive = positive;
  // blocks_start and blocks_end have the same size
  n_blocks = blocks_start->size();
  // The object is not ready (synchronize_proxs has not been called)
  is_synchronized = false;
}

template <class T, class K>
TProxWithGroups<T, K>::TProxWithGroups(K strength,
                               SArrayULongPtr blocks_start,
                               SArrayULongPtr blocks_length,
                               ulong start,
                               ulong end, bool positive)
    : TProx<K, K>(strength, start, end, positive) {
  this->blocks_start = blocks_start;
  this->blocks_length = blocks_length;
  this->positive = positive;
  // blocks_start and blocks_end have the same size
  n_blocks = blocks_start->size();
  // The object is not ready (synchronize_proxs has not been called)
  is_synchronized = false;
}

template <class T, class K>
void
TProxWithGroups<T, K>::synchronize_proxs() {
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

template <class T, class K>
std::unique_ptr<TProx<T, K> >
TProxWithGroups<T, K>::build_prox(K strength, ulong start, ulong end, bool positive) {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

template <class T, class K>
std::string
TProxWithGroups<T, K>::get_class_name() const {
  return "TProxWithGroups";
}

template <class T, class K>
K
TProxWithGroups<T, K>::value(const ArrayDouble &coeffs,
                             ulong start,
                             ulong end) {
  if (!is_synchronized) {
    synchronize_proxs();
  }
  K val = 0.;
  for (auto &prox : proxs) {
    val += prox->value(coeffs, prox->get_start(), prox->get_end());
  }
  return val;
}

template <class T, class K>
void
TProxWithGroups<T, K>::call(const ArrayDouble &coeffs,
                          K step,
                          ArrayDouble &out,
                          ulong start,
                          ulong end) {
  if (!is_synchronized) {
    synchronize_proxs();
  }
  for (auto &prox : proxs) {
    prox->call(coeffs, step, out, prox->get_start(), prox->get_end());
  }
}

template class TProxWithGroups<double, double>;
