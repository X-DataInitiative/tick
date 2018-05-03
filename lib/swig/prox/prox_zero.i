// License: BSD 3 clause

%{
#include "tick/prox/prox_zero.h"
%}

template <class T>
class TProxZero : public TProx<T> {
 public:
   TProxZero();
   TProxZero(T strength);

   TProxZero(T strength,
            unsigned long start,
            unsigned long end);

  bool compare(const TProxZero<T> &that);
};

%template(ProxZeroDouble) TProxZero<double>;
typedef TProxZero<double> ProxZeroDouble;
TICK_MAKE_TEMPLATED_PICKLABLE(TProxZero, ProxZeroDouble, double);

%template(ProxZeroFloat) TProxZero<float>;
typedef TProxZero<float> ProxZeroFloat;
TICK_MAKE_TEMPLATED_PICKLABLE(TProxZero, ProxZeroFloat , float);
