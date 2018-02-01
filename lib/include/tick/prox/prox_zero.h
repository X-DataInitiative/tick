
#ifndef LIB_INCLUDE_TICK_PROX_PROX_ZERO_H_
#define LIB_INCLUDE_TICK_PROX_PROX_ZERO_H_

// License: BSD 3 clause

#include "prox_separable.h"

template <class T, class K = T>
class DLL_PUBLIC TProxZero : public TProxSeparable<T, K> {
 public:
  explicit TProxZero(K strength);

  TProxZero(K strength,
           ulong start,
           ulong end);

  virtual std::string get_class_name() const;

  K value(const Array<T> &coeffs, ulong start, ulong end) override;

 private:
  K call_single(K x, K step) const override;

  K call_single(K x, K step, ulong n_times) const override;
};

class DLL_PUBLIC ProxZero : public TProxZero<double, double> {
 public:
  explicit ProxZero(double strength);

  ProxZero(double strength,
           ulong start,
           ulong end);

  std::string get_class_name() const override;
};

#endif  // LIB_INCLUDE_TICK_PROX_PROX_ZERO_H_
