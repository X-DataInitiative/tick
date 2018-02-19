#ifndef LIB_INCLUDE_TICK_PROX_PROX_POSITIVE_H_
#define LIB_INCLUDE_TICK_PROX_PROX_POSITIVE_H_

// License: BSD 3 clause

#include "prox_separable.h"

template <class T>
class DLL_PUBLIC TProxPositive : public TProxSeparable<T> {
 public:
  explicit TProxPositive(T strength);

  TProxPositive(T strength, ulong start, ulong end);

  std::string get_class_name() const override;

  // Override value, only this value method should be called
  T value(const Array<T> &coeffs, ulong start, ulong end) override;

 private:
  T call_single(T x, T step) const override;

  // Repeat n_times the prox on coordinate i
  T call_single(T x, T step, ulong n_times) const override;
};

using ProxPositive = TProxPositive<double>;

using ProxPositiveDouble = TProxPositive<double>;
using ProxPositiveFloat = TProxPositive<float>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_POSITIVE_H_
