// License: BSD 3 clause

%{
#include "tick/prox/prox_elasticnet.h"
%}

template <class T, class K>
class TProxElasticNet : public TProxSeparable<T, K> {

 public:
 TProxElasticNet(){}
  TProxElasticNet(T strength, T ratio, bool positive);

  TProxElasticNet(T strength, T ratio, ulong start, ulong end, bool positive);

  virtual T get_ratio() const;

  virtual void set_ratio(T ratio);

  bool compare(const TProxElasticNet<T, K> &that);
};

%template(ProxElasticNetDouble) TProxElasticNet<double, double>;
typedef TProxElasticNet<double, double> ProxElasticNetDouble;

%template(ProxElasticNetFloat) TProxElasticNet<float, float>;
typedef TProxElasticNet<float, float> ProxElasticNetFloat;
