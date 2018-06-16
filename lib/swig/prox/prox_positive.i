// License: BSD 3 clause

%{
#include "tick/prox/prox_positive.h"
%}

template <class T, class K>
class TProxPositive : public TProx<T, K> {
 public:
   TProxPositive(T strength);

   TProxPositive(T strength,
          unsigned long start,
          unsigned long end);

  bool compare(const TProxPositive<T, K> &that);
};

%template(ProxPositiveDouble) TProxPositive<double, double>;
typedef TProxPositive<double, double> ProxPositiveDouble;

%template(ProxPositiveFloat) TProxPositive<float, float>;
typedef TProxPositive<float, float> ProxPositiveFloat;
