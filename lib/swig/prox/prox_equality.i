// License: BSD 3 clause

%{
#include "tick/prox/prox_equality.h"
%}

template <class T>
class TProxEquality : public TProx<T> {
 public:
  explicit TProxEquality(T strength, bool positive);

  TProxEquality(T strength, ulong start, ulong end, bool positive);
};

%template(ProxEquality) TProxEquality<double>;
typedef TProxEquality<double> ProxEquality;

%template(ProxEqualityDouble) TProxEquality<double>;
typedef TProxEquality<double> ProxEqualityDouble;

%template(ProxEqualityFloat) TProxEquality<float>;
typedef TProxEquality<float> ProxEqualityFloat;
