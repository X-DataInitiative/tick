// License: BSD 3 clause

%{
#include "tick/prox/prox_elasticnet.h"
%}

template <class T>
class TProxElasticNet : public TProxSeparable<T> {

 public:
 TProxElasticNet(){}
  TProxElasticNet(T strength, T ratio, bool positive);

  TProxElasticNet(T strength, T ratio, ulong start, ulong end, bool positive);

  virtual T get_ratio() const;

  virtual void set_ratio(T ratio);

  bool compare(const TProxElasticNet<T> &that);
};

%template(ProxElasticNetDouble) TProxElasticNet<double>;
typedef TProxElasticNet<double> ProxElasticNetDouble;
TICK_MAKE_TEMPLATED_PICKLABLE(TProxElasticNet, ProxElasticNetDouble , double);

%template(ProxElasticNetFloat) TProxElasticNet<float>;
typedef TProxElasticNet<float> ProxElasticNetFloat;
TICK_MAKE_TEMPLATED_PICKLABLE(TProxElasticNet, ProxElasticNetFloat , float);
