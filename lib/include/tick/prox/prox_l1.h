#ifndef LIB_INCLUDE_TICK_PROX_PROX_L1_H_
#define LIB_INCLUDE_TICK_PROX_PROX_L1_H_

// License: BSD 3 clause

#include "prox_separable.h"

template <class T>
class DLL_PUBLIC TProxL1 : public TProxSeparable<T> {
 protected:
  using TProx<T>::has_range;
  using TProx<T>::strength;
  using TProx<T>::start;
  using TProx<T>::end;
  using TProx<T>::positive;

 public:
  using TProxSeparable<T>::get_class_name;

 public:
  TProxL1(T strength, bool positive);

  TProxL1(T strength, ulong start, ulong end, bool positive);

 protected:
  T call_single(T x, T step) const override;

  // Repeat n_times the prox on coordinate i
  T call_single(T x, T step, ulong n_times) const override;

  T value_single(T x) const override;
};

using ProxL1 = TProxL1<double>;

using ProxL1Double = TProxL1<double>;
using ProxL1Float = TProxL1<float>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_L1_H_
