// License: BSD 3 clause

%{
#include "tick/prox/prox_positive.h"
%}

template <class T>
class TProxPositive : public TProx<T> {
 public:
   TProxPositive(T strength);

   TProxPositive(T strength,
          unsigned long start,
          unsigned long end);

  bool compare(const TProxPositive<T> &that);
};

%template(ProxPositive) TProxPositive<double>;
typedef TProxPositive<double> ProxPositive;

%template(ProxPositiveDouble) TProxPositive<double>;
typedef TProxPositive<double> ProxPositiveDouble;

%template(ProxPositiveFloat) TProxPositive<float>;
typedef TProxPositive<float> ProxPositiveFloat;
