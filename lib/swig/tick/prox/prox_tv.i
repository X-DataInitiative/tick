// License: BSD 3 clause

%{
#include "tick/prox/prox_tv.h"
%}

template <class T, class K>
class TProxTV : public TProx<T, K> {
 public:
   TProxTV(T strength,
             bool positive);

   TProxTV(T strength,
             unsigned long start,
             unsigned long end,
             bool positive);

  bool compare(const TProxTV<T, K> &that);
};

%template(ProxTVDouble) TProxTV<double, double>;
typedef TProxTV<double, double> ProxTVDouble;
TICK_MAKE_TEMPLATED_PICKLABLE(TProxTV, ProxTVDouble , double);

%template(ProxTVFloat) TProxTV<float, float>;
typedef TProxTV<float, float> ProxTVFloat;
TICK_MAKE_TEMPLATED_PICKLABLE(TProxTV, ProxTVFloat , float);

