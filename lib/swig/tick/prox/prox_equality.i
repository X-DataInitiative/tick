// License: BSD 3 clause

%{
#include "tick/prox/prox_equality.h"
%}

template <class T, class K>
class TProxEquality : public TProx<T, K> {
 public:
  TProxEquality();
  explicit TProxEquality(T strength, bool positive);

  TProxEquality(T strength, ulong start, ulong end, bool positive);

  bool compare(const TProxEquality<T, K> &that);
};

%template(ProxEqualityDouble) TProxEquality<double, double>;
typedef TProxEquality<double, double> ProxEqualityDouble;
TICK_MAKE_TK_PICKLABLE(TProxEquality, ProxEqualityDouble ,  double,  double);

%template(ProxEqualityFloat) TProxEquality<float, float>;
typedef TProxEquality<float, float> ProxEqualityFloat;
TICK_MAKE_TK_PICKLABLE(TProxEquality, ProxEqualityFloat ,  float,  float);
