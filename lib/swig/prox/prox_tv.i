// License: BSD 3 clause

%{
#include "tick/prox/prox_tv.h"
%}

template <class T>
class TProxTV : public TProx<T> {
 public:
   TProxTV(T strength,
             bool positive);

   TProxTV(T strength,
             unsigned long start,
             unsigned long end,
             bool positive);

  bool compare(const TProxTV<T> &that);
};

%template(ProxTVDouble) TProxTV<double>;
typedef TProxTV<double> ProxTVDouble;
TICK_MAKE_TEMPLATED_PICKLABLE(TProxTV, ProxTVDouble , double);

%template(ProxTVFloat) TProxTV<float>;
typedef TProxTV<float> ProxTVFloat;
TICK_MAKE_TEMPLATED_PICKLABLE(TProxTV, ProxTVFloat , float);

