#ifndef LIB_INCLUDE_TICK_PROX_PROX_L1_H_
#define LIB_INCLUDE_TICK_PROX_PROX_L1_H_

// License: BSD 3 clause

#include "prox_separable.h"

template <class T, class K>
class TProxL1 : public TProxSeparable<T, K> {
 protected:
  using TProx<T, K>::has_range;
  using TProx<T, K>::strength;
  using TProx<T, K>::start;
  using TProx<T, K>::end;
  using TProx<T, K>::positive;

 public:
  TProxL1(K strength, bool positive);

  TProxL1(K strength, ulong start, ulong end, bool positive);

  virtual std::string get_class_name() const;

 protected:
  K call_single(K x, K step) const override;

  // Repeat n_times the prox on coordinate i
  K call_single(K x, K step, ulong n_times) const override;

  K value_single(K x) const override;
};


class ProxL1 : public TProxL1<double, double> {
 public:
  ProxL1(double strength, bool positive);

  ProxL1(double strength, ulong start, ulong end, bool positive);

  std::string get_class_name() const override;
};

using ProxL1Double = TProxL1<double, double>;
using ProxL1Float  = TProxL1<float , float>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_L1_H_
