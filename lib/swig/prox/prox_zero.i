// License: BSD 3 clause

%{
#include "tick/prox/prox_zero.h"
%}

template <class T>
class TProxZero : public TProx<T> {
 public:
   TProxZero(T strength);

   TProxZero(T strength,
            unsigned long start,
            unsigned long end);
};

%template(ProxZero) TProxZero<double>;
typedef TProxZero<double> ProxZero;

%template(ProxZeroDouble) TProxZero<double>;
typedef TProxZero<double> ProxZeroDouble;

%template(ProxZeroFloat) TProxZero<float>;
typedef TProxZero<float> ProxZeroFloat;

