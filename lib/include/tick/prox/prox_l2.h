#ifndef LIB_INCLUDE_TICK_PROX_PROX_L2_H_
#define LIB_INCLUDE_TICK_PROX_PROX_L2_H_

// License: BSD 3 clause

#include "prox.h"

template <class T>
class TProxL2 : public TProx<T> {
 protected:
  using TProx<T>::strength;
  using TProx<T>::positive;

 public:
  TProxL2(T strength, bool positive);

  TProxL2(T strength, ulong start, ulong end, bool positive);

  std::string get_class_name() const override;

  T value(const Array<T> &coeffs, ulong start, ulong end) override;

  void call(const Array<T> &coeffs, T step, Array<T> &out, ulong start,
            ulong end) override;
};

using ProxL2 = TProxL2<double>;

using ProxL2Double = TProxL2<double>;
using ProxL2Float = TProxL2<float>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_L2_H_
