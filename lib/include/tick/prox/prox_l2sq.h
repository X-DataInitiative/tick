#ifndef LIB_INCLUDE_TICK_PROX_PROX_L2SQ_H_
#define LIB_INCLUDE_TICK_PROX_PROX_L2SQ_H_

// License: BSD 3 clause

#include "prox_separable.h"

template <class T>
class DLL_PUBLIC TProxL2Sq : public TProxSeparable<T> {
 protected:
  using TProxSeparable<T>::has_range;
  using TProxSeparable<T>::strength;
  using TProxSeparable<T>::start;
  using TProxSeparable<T>::end;
  using TProxSeparable<T>::positive;

 public:
  TProxL2Sq(T strength, bool positive);

  TProxL2Sq(T strength, ulong start, ulong end, bool positive);

  virtual std::string get_class_name() const;

 protected:
  T value_single(T x) const override;

  T call_single(T x, T step) const override;

  // Repeat n_times the prox on coordinate i
  T call_single(T x, T step, ulong n_times) const override;
};

using ProxL2Sq = TProxL2Sq<double>;

using ProxL2SqDouble = TProxL2Sq<double>;
using ProxL2SqFloat = TProxL2Sq<float>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_L2SQ_H_
