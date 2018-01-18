#ifndef LIB_INCLUDE_TICK_PROX_PROX_ELASTICNET_H_
#define LIB_INCLUDE_TICK_PROX_PROX_ELASTICNET_H_

// License: BSD 3 clause

#include "prox_separable.h"

class ProxElasticNet : public ProxSeparable {
 protected:
  double ratio;

 public:
  ProxElasticNet(double strength, double ratio, bool positive);

  ProxElasticNet(double strength, double ratio, ulong start, ulong end, bool positive);

  const std::string get_class_name() const override;

  virtual double get_ratio() const;

  virtual void set_ratio(double ratio);

 private:
  double call_single(double x, double step) const override;

  double value_single(double x) const override;
};

#endif  // LIB_INCLUDE_TICK_PROX_PROX_ELASTICNET_H_
