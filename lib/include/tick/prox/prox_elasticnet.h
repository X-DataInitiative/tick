#ifndef LIB_INCLUDE_TICK_PROX_PROX_ELASTICNET_H_
#define LIB_INCLUDE_TICK_PROX_PROX_ELASTICNET_H_

// License: BSD 3 clause

#include "prox_separable.h"

template <class T>
class DLL_PUBLIC TProxElasticNet : public TProxSeparable<T> {
 protected:
  using TProxSeparable<T>::strength;
  using TProxSeparable<T>::positive;

 protected:
  T ratio;

 public:
  TProxElasticNet(T strength, T ratio, bool positive);

  TProxElasticNet(T strength, T ratio, ulong start, ulong end, bool positive);

  std::string get_class_name() const override;

  virtual T get_ratio() const;

  virtual void set_ratio(T ratio);

 private:
  T call_single(T x, T step) const override;

  T value_single(T x) const override;
};

using ProxElasticNet = TProxElasticNet<double>;

using ProxElasticNetDouble = TProxElasticNet<double>;
using ProxElasticNetFloat = TProxElasticNet<float>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_ELASTICNET_H_
