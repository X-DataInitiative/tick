
#ifndef LIB_INCLUDE_TICK_PROX_PROX_MULTI_H_
#define LIB_INCLUDE_TICK_PROX_PROX_MULTI_H_

// License: BSD 3 clause

#include "prox.h"

// TODO: this requires some work. ProxMulti should have the standard
// TODO: prox API, with a set_strength, and things like that

template <class T>
class DLL_PUBLIC TProxMulti : public TProx<T> {
  using ProxTPtrVector = std::vector<std::shared_ptr<TProx<T> > >;

 protected:
  ProxTPtrVector proxs;

 public:
  explicit TProxMulti(ProxTPtrVector proxs);

  std::string get_class_name() const override;

  T value(const Array<T> &coeffs, ulong start, ulong end) override;

  void call(const Array<T> &coeffs, T step, Array<T> &out, ulong start,
            ulong end) override;
};

using ProxMulti = TProxMulti<double>;
using ProxMultiDouble = TProxMulti<double>;
using ProxMultiFloat = TProxMulti<float>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_MULTI_H_
