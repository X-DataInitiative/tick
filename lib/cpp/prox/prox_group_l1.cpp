// License: BSD 3 clause

#include "tick/prox/prox_group_l1.h"
#include "tick/prox/prox_l2.h"

template <class T>
std::unique_ptr<TProx<T> > TProxGroupL1<T>::build_prox(T strength, ulong start,
                                                       ulong end,
                                                       bool positive) {
  return std::move(std::unique_ptr<TProxL2<T> >(
      new TProxL2<T>(strength, start, end, positive)));
}

template class DLL_PUBLIC TProxGroupL1<double>;
template class DLL_PUBLIC TProxGroupL1<float>;
