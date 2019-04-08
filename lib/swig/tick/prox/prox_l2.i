// License: BSD 3 clause

%{
#include "tick/prox/prox_l2.h"
%}

template <class T, class K>
class TProxL2 : public TProx<T, K> {
 public:
   TProxL2();
   TProxL2(T strength,
          bool positive);

   TProxL2(T strength,
          unsigned long start,
          unsigned long end,
          bool positive);

  bool compare(const TProxL2<T, K> &that);
};

%template(ProxL2Double) TProxL2<double, double>;
typedef TProxL2<double, double> ProxL2Double;
TICK_MAKE_TK_PICKLABLE(TProxL2, ProxL2Double ,  double,  double);

%template(ProxL2Float) TProxL2<float, float>;
typedef TProxL2<float, float> ProxL2Float;
TICK_MAKE_TK_PICKLABLE(TProxL2, ProxL2Float ,  float,  float);
