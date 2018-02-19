#ifndef LIB_INCLUDE_TICK_PROX_PROX_EQUALITY_H_
#define LIB_INCLUDE_TICK_PROX_PROX_EQUALITY_H_

// License: BSD 3 clause

#include "prox.h"

template <class T>
class DLL_PUBLIC TProxEquality : public TProx<T> {
 protected:
  using TProx<T>::positive;

 public:
  explicit TProxEquality(T strength, bool positive);

  TProxEquality(T strength, ulong start, ulong end, bool positive);

  std::string get_class_name() const override;

  T value(const Array<T> &coeffs, ulong start, ulong end) override;

  void call(const Array<T> &coeffs, T step, Array<T> &out, ulong start,
            ulong end) override;
};

using ProxEquality = TProxEquality<double>;

using ProxEqualityDouble = TProxEquality<double>;
using ProxEqualityFloat = TProxEquality<float>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_EQUALITY_H_
