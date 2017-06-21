#ifndef TICK_OPTIM_PROX_SRC_PROX_ELASTICNET_H_
#define TICK_OPTIM_PROX_SRC_PROX_ELASTICNET_H_

#include "prox_separable.h"

class ProxElasticNet : public ProxSeparable {
 protected:
  double ratio;

 public:
  ProxElasticNet(double strength, double ratio, bool positive);

  ProxElasticNet(double strength, double ratio, ulong start, ulong end, bool positive);

  const std::string get_class_name() const override;

  double call_single(double x, double step) const override;

  double value_single(double x) const override;

  virtual double get_ratio() const;

  virtual void set_ratio(double ratio);
};

#endif  // TICK_OPTIM_PROX_SRC_PROX_ELASTICNET_H_
