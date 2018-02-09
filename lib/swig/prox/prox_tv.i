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
};

%template(ProxTV) TProxTV<double>;
typedef TProxTV<double> ProxTV;

%template(ProxTVDouble) TProxTV<double>;
typedef TProxTV<double> ProxTVDouble;

%template(ProxTVFloat) TProxTV<float>;
typedef TProxTV<float> ProxTVFloat;

