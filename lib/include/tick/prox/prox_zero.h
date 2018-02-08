
#ifndef LIB_INCLUDE_TICK_PROX_PROX_ZERO_H_
#define LIB_INCLUDE_TICK_PROX_PROX_ZERO_H_

// License: BSD 3 clause

#include "prox_separable.h"

template <class T>
class DLL_PUBLIC TProxZero : public TProxSeparable<T> {
 public:
  explicit TProxZero(T strength);

  TProxZero(T strength, ulong start, ulong end);

  std::string get_class_name() const override;

  T value(const Array<T> &coeffs, ulong start, ulong end) override;

 private:
  T call_single(T x, T step) const override;

  T call_single(T x, T step, ulong n_times) const override;
};

using ProxZero = TProxZero<double>;

using ProxZeroDouble = TProxZero<double>;
using ProxZeroFloat = TProxZero<float>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_ZERO_H_
