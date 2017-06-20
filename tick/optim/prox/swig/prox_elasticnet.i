%{
#include "prox_elasticnet.h"
%}

class ProxElasticNet : public ProxSeparable {
 public:
  ProxElasticNet(double strength,
                 double ratio,
                 bool positive);

  ProxElasticNet(double strength,
                 double ratio,
                 ulong start,
                 ulong end,
                 bool positive);

  virtual double get_ratio() const final;

  virtual void set_ratio(double ratio) final;
};
