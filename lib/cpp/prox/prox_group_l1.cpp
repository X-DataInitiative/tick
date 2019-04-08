// License: BSD 3 clause

#include "tick/prox/prox_group_l1.h"
#include "tick/prox/prox_l2.h"

template <class T, class K>
std::unique_ptr<TProx<T, K>> TProxGroupL1<T, K>::build_prox(T strength,
                                                            ulong start,
                                                            ulong end,
                                                            bool positive) {
  return std::move(std::unique_ptr<TProxL2<T, K>>(
      new TProxL2<T, K>(strength, start, end, positive)));
}

template class DLL_PUBLIC TProxGroupL1<double, double>;
template class DLL_PUBLIC TProxGroupL1<float, float>;

template class DLL_PUBLIC TProxGroupL1<double, std::atomic<double>>;
template class DLL_PUBLIC TProxGroupL1<float, std::atomic<float>>;
