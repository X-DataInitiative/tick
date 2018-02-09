// License: BSD 3 clause

#include "tick/prox/prox_group_l1.h"
#include "tick/prox/prox_l2.h"

template <class T>
TProxGroupL1<T>::TProxGroupL1(T strength, SArrayULongPtr blocks_start,
                              SArrayULongPtr blocks_length, bool positive)
    : TProxWithGroups<T>(strength, blocks_start, blocks_length, positive) {}

template <class T>
TProxGroupL1<T>::TProxGroupL1(T strength, SArrayULongPtr blocks_start,
                              SArrayULongPtr blocks_length, ulong start,
                              ulong end, bool positive)
    : TProxWithGroups<T>(strength, blocks_start, blocks_length, start, end,
                         positive) {}

template <class T>
std::string TProxGroupL1<T>::get_class_name() const {
  return "TProxGroupL1";
}

template <class T>
std::unique_ptr<TProx<T> > TProxGroupL1<T>::build_prox(T strength, ulong start,
                                                       ulong end,
                                                       bool positive) {
  return std::unique_ptr<TProxL2<T> >(
      new TProxL2<T>(strength, start, end, positive));
}

template class TProxGroupL1<double>;
template class TProxGroupL1<float>;
