// License: BSD 3 clause

%{
#include "tick/prox/prox_l2.h"
%}

template <class T>
class TProxL2 : public TProx<T> {
 public:
   TProxL2(T strength,
          bool positive);

   TProxL2(T strength,
          unsigned long start,
          unsigned long end,
          bool positive);

  bool compare(const TProxL2<T> &that);
};

%template(ProxL2) TProxL2<double>;
typedef TProxL2<double> ProxL2;

%template(ProxL2Double) TProxL2<double>;
typedef TProxL2<double> ProxL2Double;

%template(ProxL2Float) TProxL2<float>;
typedef TProxL2<float> ProxL2Float;
