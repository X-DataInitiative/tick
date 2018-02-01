#ifndef LIB_INCLUDE_TICK_PROX_PROX_L2SQ_H_
#define LIB_INCLUDE_TICK_PROX_PROX_L2SQ_H_

// License: BSD 3 clause

#include "prox_separable.h"

template <class T, class K>
class TProxL2Sq : public TProxSeparable<T, K> {
 protected:
  using TProxSeparable<T, K>::has_range;
  using TProxSeparable<T, K>::strength;
  using TProxSeparable<T, K>::start;
  using TProxSeparable<T, K>::end;
  using TProxSeparable<T, K>::positive;

 public:
  TProxL2Sq(K strength,
            bool positive);

  TProxL2Sq(K strength,
            ulong start,
            ulong end,
            bool positive);

  virtual std::string get_class_name() const;

 protected:
  K value_single(K x) const override;

  K call_single(K x, K step) const override;

  // Repeat n_times the prox on coordinate i
  K call_single(K x, K step, ulong n_times) const override;
};

class ProxL2Sq : public TProxL2Sq<double, double> {
 public:
  ProxL2Sq(double strength, bool positive);

  ProxL2Sq(double strength, ulong start, ulong end, bool positive);

  std::string get_class_name() const override;
};

using ProxL2SqDouble = TProxL2Sq<double, double>;
using ProxL2SqFloat  = TProxL2Sq<float , float>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_L2SQ_H_
