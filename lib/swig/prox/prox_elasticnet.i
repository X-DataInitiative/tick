// License: BSD 3 clause

%{
#include "tick/prox/prox_elasticnet.h"
%}

template <class T>
class TProxElasticNet : public TProxSeparable<T> {

 public:
  TProxElasticNet(T strength, T ratio, bool positive);

  TProxElasticNet(T strength, T ratio, ulong start, ulong end, bool positive);

  virtual T get_ratio() const;

  virtual void set_ratio(T ratio);
};

%template(ProxElasticNet) TProxElasticNet<double>;
typedef TProxElasticNet<double> ProxElasticNet;

%template(ProxElasticNetDouble) TProxElasticNet<double>;
typedef TProxElasticNet<double> ProxElasticNetDouble;

%template(ProxElasticNetFloat) TProxElasticNet<float>;
typedef TProxElasticNet<float> ProxElasticNetFloat;
